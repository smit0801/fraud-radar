"""
Replay transactions from CSV into a Kafka topic as if they were live traffic.

Speed control: --speedup 3600 compresses 1 hour of real Time deltas into 1 second.
Use --rate to just fire N transactions/sec and ignore original timing.

Usage:
    python streaming/producer.py --data data/transactions.csv --topic transactions --rate 50
"""
import argparse
import json
import time

import pandas as pd
from kafka import KafkaProducer

FEATURES = [f"V{i}" for i in range(1, 29)]


def main(data: str, topic: str, bootstrap: str, rate: float, limit: int):
    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode(),
        key_serializer=lambda k: k.encode(),
        linger_ms=5,
    )
    df = pd.read_csv(data)
    if limit:
        df = df.head(limit)

    interval = 1.0 / rate
    sent = 0
    print(f"Replaying {len(df):,} transactions -> topic '{topic}' at {rate}/s ...")
    for i, row in df.iterrows():
        msg = {
            "transaction_id": f"txn_{i:07d}",
            "features": [row[f] for f in FEATURES],
            "amount": row["Amount"],
            "true_label": int(row["Class"]),  # kept for offline evaluation only
            "produced_at": time.time(),
        }
        producer.send(topic, key=msg["transaction_id"], value=msg)
        sent += 1
        if sent % 500 == 0:
            print(f"  sent {sent:,}")
        time.sleep(interval)

    producer.flush()
    print(f"Done. {sent:,} transactions sent.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/transactions.csv")
    p.add_argument("--topic", default="transactions")
    p.add_argument("--bootstrap", default="localhost:9092")
    p.add_argument("--rate", type=float, default=50, help="transactions per second")
    p.add_argument("--limit", type=int, default=0, help="max rows to send (0 = all)")
    args = p.parse_args()
    main(args.data, args.topic, args.bootstrap, args.rate, args.limit)
