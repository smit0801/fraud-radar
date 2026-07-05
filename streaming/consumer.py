"""
Consume transactions from Kafka, score them via the FastAPI service in
micro-batches, and publish anything >= review threshold to an 'alerts' topic.

Micro-batching matters: calling /score once per message caps you at
~1/latency msgs/sec. Batching 100 at a time via /score/batch keeps
end-to-end latency low while multiplying throughput ~50x.

Usage (API must be running on :8000):
    python streaming/consumer.py --topic transactions --batch-size 100
"""
import argparse
import json
import time

import httpx
from kafka import KafkaConsumer, KafkaProducer


def main(topic: str, bootstrap: str, api_url: str, batch_size: int, alert_topic: str):
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        value_deserializer=lambda v: json.loads(v.decode()),
        auto_offset_reset="earliest",
        group_id="fraud-scorer",
        max_poll_records=batch_size,
    )
    alert_producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode(),
    )
    client = httpx.Client(base_url=api_url, timeout=10.0)

    stats = {"scored": 0, "review": 0, "block": 0, "tp": 0, "fp": 0}
    print(f"Consuming '{topic}', scoring via {api_url} in batches of {batch_size} ...")

    while True:
        records = consumer.poll(timeout_ms=1000, max_records=batch_size)
        batch = [msg.value for msgs in records.values() for msg in msgs]
        if not batch:
            continue

        payload = [
            {"transaction_id": m["transaction_id"], "features": m["features"], "amount": m["amount"]}
            for m in batch
        ]
        t0 = time.perf_counter()
        resp = client.post("/score/batch", json=payload)
        resp.raise_for_status()
        results = resp.json()
        ms = (time.perf_counter() - t0) * 1000

        labels = {m["transaction_id"]: m.get("true_label", 0) for m in batch}
        for r in results:
            stats["scored"] += 1
            if r["decision"] in ("review", "block"):
                stats[r["decision"]] += 1
                is_fraud = labels.get(r["transaction_id"], 0) == 1
                stats["tp" if is_fraud else "fp"] += 1
                alert_producer.send(alert_topic, value={**r, "true_label": labels.get(r["transaction_id"])})
                flag = "FRAUD" if is_fraud else "clean"
                print(f"  ALERT {r['transaction_id']} risk={r['risk_score']:.1f} "
                      f"decision={r['decision']} (actually {flag})")

        print(f"batch={len(batch)} scored in {ms:.0f}ms | "
              f"total={stats['scored']:,} review={stats['review']} block={stats['block']} "
              f"tp={stats['tp']} fp={stats['fp']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--topic", default="transactions")
    p.add_argument("--bootstrap", default="localhost:9092")
    p.add_argument("--api-url", default="http://localhost:8000")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--alert-topic", default="alerts")
    args = p.parse_args()
    main(args.topic, args.bootstrap, args.api_url, args.batch_size, args.alert_topic)
