"""
Generate a synthetic dataset matching the Kaggle credit card fraud schema:
    Time, V1..V28, Amount, Class

The real dataset's V1-V28 are PCA components of anonymized features.
We mimic that statistically: normal transactions ~ N(0, 1) per component,
fraud transactions are shifted/scaled in a random subset of components,
which is exactly the kind of structure Isolation Forest exploits.

Swap in the real creditcard.csv from Kaggle later — everything downstream
(training, API, replay) reads the same schema and won't need changes.

Usage:
    python data/generate_synthetic.py --rows 50000 --fraud-rate 0.0017 --out data/transactions.csv
"""
import argparse

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def make_dataset(n_rows: int, fraud_rate: float) -> pd.DataFrame:
    n_fraud = max(1, int(n_rows * fraud_rate))
    n_normal = n_rows - n_fraud

    # --- normal transactions: standard normal PCA components
    normal_v = RNG.normal(0.0, 1.0, size=(n_normal, 28))
    # log-normal amounts, mostly small purchases
    normal_amt = np.round(RNG.lognormal(mean=3.0, sigma=1.2, size=n_normal), 2)

    # --- fraud: shift a random subset of components (mimics real dataset,
    # where V3, V4, V10, V12, V14, V17 separate fraud strongly)
    fraud_v = RNG.normal(0.0, 1.0, size=(n_fraud, 28))
    hot_components = [2, 3, 9, 11, 13, 16]  # 0-indexed: V3,V4,V10,V12,V14,V17
    for c in hot_components:
        sign = -1 if c in (2, 9, 11, 13, 16) else 1
        fraud_v[:, c] += sign * RNG.normal(4.0, 1.5, size=n_fraud)
    # fraud amounts: bimodal — card testing (tiny) and cash-out (large)
    small = np.round(RNG.uniform(0.5, 5.0, size=n_fraud), 2)
    large = np.round(RNG.lognormal(mean=5.5, sigma=0.8, size=n_fraud), 2)
    fraud_amt = np.where(RNG.random(n_fraud) < 0.4, small, large)

    v_cols = [f"V{i}" for i in range(1, 29)]
    df_normal = pd.DataFrame(normal_v, columns=v_cols)
    df_normal["Amount"] = normal_amt
    df_normal["Class"] = 0

    df_fraud = pd.DataFrame(fraud_v, columns=v_cols)
    df_fraud["Amount"] = fraud_amt
    df_fraud["Class"] = 1

    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    # seconds elapsed, like the real dataset (~2 days of traffic)
    df.insert(0, "Time", np.sort(RNG.uniform(0, 172_800, size=len(df))).round(1))
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=50_000)
    p.add_argument("--fraud-rate", type=float, default=0.0017)  # matches real dataset (~0.17%)
    p.add_argument("--out", type=str, default="data/transactions.csv")
    args = p.parse_args()

    df = make_dataset(args.rows, args.fraud_rate)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows ({df['Class'].sum()} fraud) -> {args.out}")
