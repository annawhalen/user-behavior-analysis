"""
generate_data.py
----------------
Generates realistic simulated marketplace event and order data.
Run this first to create the datasets used by all other modules.

Output files:
    data/marketplace_events.csv   — user-level event stream
    data/orders.csv               — completed order records
    data/users.csv                — user profiles
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

N_USERS = 15_000
N_DAYS = 180
START_DATE = datetime(2024, 1, 1)

TRAFFIC_SOURCES = ["organic", "paid_search", "social", "email", "direct", "referral"]
TRAFFIC_WEIGHTS = [0.30, 0.25, 0.20, 0.10, 0.10, 0.05]
DEVICE_TYPES = ["mobile", "desktop", "tablet"]
DEVICE_WEIGHTS = [0.60, 0.32, 0.08]
CATEGORIES = ["trading_cards", "fashion", "electronics", "collectibles", "live_plants", "sports"]

EVENT_FUNNEL = [
    "page_view", "listing_view", "signup_start", "signup_complete",
    "add_to_cart", "checkout_start", "purchase_complete"
]

DROP_RATES = [0.0, 0.45, 0.60, 0.30, 0.40, 0.35, 0.45]


def generate_users(n: int) -> pd.DataFrame:
    signup_dates = [START_DATE + timedelta(days=random.randint(0, N_DAYS - 30)) for _ in range(n)]
    return pd.DataFrame({
        "user_id": [f"u_{i:06d}" for i in range(n)],
        "signup_date": signup_dates,
        "country": np.random.choice(["US", "UK", "CA", "AU"], n, p=[0.70, 0.15, 0.10, 0.05]),
        "device_type": np.random.choice(DEVICE_TYPES, n, p=DEVICE_WEIGHTS),
        "acquisition_source": np.random.choice(TRAFFIC_SOURCES, n, p=TRAFFIC_WEIGHTS),
    })


def generate_events(users: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, user in users.iterrows():
        n_sessions = np.random.poisson(lam=4)
        for _ in range(max(1, n_sessions)):
            session_date = user["signup_date"] + timedelta(days=random.randint(0, 60))
            if session_date > START_DATE + timedelta(days=N_DAYS):
                continue
            session_id = f"s_{random.randint(100000, 999999)}"
            ts = session_date + timedelta(seconds=random.randint(0, 86400))

            for i, event_type in enumerate(EVENT_FUNNEL):
                if i > 0 and random.random() < DROP_RATES[i]:
                    break
                records.append({
                    "event_id": f"e_{random.randint(1000000, 9999999)}",
                    "user_id": user["user_id"],
                    "session_id": session_id,
                    "event_type": event_type,
                    "event_timestamp": ts + timedelta(seconds=i * random.randint(10, 120)),
                    "device_type": user["device_type"],
                    "traffic_source": user["acquisition_source"],
                    "utm_campaign": f"camp_{random.randint(1, 20)}",
                    "category": random.choice(CATEGORIES),
                })
    return pd.DataFrame(records)


def generate_orders(events: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    purchasers = events[events["event_type"] == "purchase_complete"]["user_id"].unique()
    records = []
    seller_pool = users["user_id"].sample(frac=0.15).tolist()

    for user_id in purchasers:
        n_orders = np.random.poisson(lam=2) + 1
        for _ in range(n_orders):
            order_date = START_DATE + timedelta(days=random.randint(0, N_DAYS))
            gmv = round(np.random.lognormal(mean=3.5, sigma=0.8), 2)
            records.append({
                "order_id": f"o_{random.randint(1000000, 9999999)}",
                "buyer_id": user_id,
                "seller_id": random.choice(seller_pool),
                "created_at": order_date,
                "gmv": gmv,
                "platform_fee": round(gmv * 0.08, 2),
                "status": np.random.choice(["completed", "refunded"], p=[0.95, 0.05]),
                "category": random.choice(CATEGORIES),
                "seller_rating": round(np.random.normal(4.4, 0.4), 1),
            })
    return pd.DataFrame(records)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("Generating users...")
    users = generate_users(N_USERS)
    users.to_csv("data/users.csv", index=False)

    print("Generating events...")
    events = generate_events(users)
    events.to_csv("data/marketplace_events.csv", index=False)

    print("Generating orders...")
    orders = generate_orders(events, users)
    orders.to_csv("data/orders.csv", index=False)

    print(f"\nDone.")
    print(f"  Users:  {len(users):,}")
    print(f"  Events: {len(events):,}")
    print(f"  Orders: {len(orders):,}")
