import pandas as pd
from sqlalchemy import create_engine, text

import feast


def verify_parity():
    print("Verifying Training-Serving Feature Parity...")

    # 1. Query Offline Store (Postgres) directly via SQL
    engine = create_engine("postgresql://fti_user:fti_password@localhost:5432/fti_db")
    query = text("""
        SELECT store_id, event_timestamp, rolling_7d_avg_sales 
        FROM f_daily_store_sales 
        WHERE store_id = 1 AND rolling_7d_avg_sales IS NOT NULL
        ORDER BY event_timestamp DESC 
        LIMIT 1
    """)
    with engine.connect() as conn:
        offline_df = pd.read_sql(query, conn)

    if offline_df.empty:
        print("No data found in offline store.")
        return

    target_store = int(offline_df.iloc[0]["store_id"])
    target_time = offline_df.iloc[0]["event_timestamp"]
    offline_val = offline_df.iloc[0]["rolling_7d_avg_sales"]

    print(
        f"Offline (Postgres) value for store {target_store} at {target_time}: {offline_val}"
    )

    # 2. Query Online Store (Redis) via Feast SDK
    fs = feast.FeatureStore(repo_path="feast/")

    # Feast online retrieval requires the exact timestamp format
    online_features = fs.get_online_features(
        features=["store_sales_features:rolling_7d_avg_sales"],
        entity_rows=[{"store_id": target_store}],
    ).to_dict()

    online_val = online_features["rolling_7d_avg_sales"][0]
    print(f"Online (Redis) latest value for store {target_store}: {online_val}")

    # 3. Assert Parity
    # Note: Floating point math can sometimes cause micro-drift, so we use a tiny tolerance
    if abs(offline_val - online_val) < 1e-5:
        print(
            "\n✅ SUCCESS: Offline and Online features match perfectly. FTI Contract enforced!"
        )
    else:
        print(
            f"\n❌ FAILURE: Mismatch detected! Offline: {offline_val}, Online: {online_val}"
        )


if __name__ == "__main__":
    verify_parity()
