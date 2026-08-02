"""
Proves the FTI contract is unbroken.
Asserts that the exact same feature definition resolves to the exact same value
in both the Offline (Postgres) and Online (Redis) stores.
"""

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

import feast


@pytest.fixture(scope="module")
def feast_store():
    return feast.FeatureStore(repo_path="feast/")


@pytest.fixture(scope="module")
def latest_entity():
    """Gets the most recent store_id and timestamp from the offline store."""
    engine = create_engine("postgresql://fti_user:fti_password@localhost:5432/fti_db")
    with engine.connect() as conn:
        # We pick the MAX timestamp because the online store only holds the latest state
        query = text("""
            SELECT store_id, event_timestamp 
            FROM f_daily_store_sales 
            WHERE rolling_7d_avg_sales IS NOT NULL 
            ORDER BY event_timestamp DESC, store_id ASC 
            LIMIT 1
        """)
        df = pd.read_sql(query, conn)
    return int(df.iloc[0]["store_id"]), df.iloc[0]["event_timestamp"]


def test_training_serving_feature_parity(feast_store, latest_entity):
    """
    The ultimate FTI proof: Offline and Online must match perfectly.
    """
    store_id, target_ts = latest_entity
    feature_ref = "store_sales_features:rolling_7d_avg_sales"

    # 1. Query Offline Store (Point-in-Time)
    entity_df = pd.DataFrame({"store_id": [store_id], "event_timestamp": [target_ts]})
    offline_df = feast_store.get_historical_features(
        entity_df=entity_df, features=[feature_ref]
    ).to_df()
    offline_val = offline_df.iloc[0]["rolling_7d_avg_sales"]

    # 2. Query Online Store (Latest State)
    online_df = feast_store.get_online_features(
        features=[feature_ref], entity_rows=[{"store_id": store_id}]
    ).to_df()
    online_val = online_df.iloc[0]["rolling_7d_avg_sales"]

    # 3. Assert Parity (allowing for microscopic floating point drift)
    assert offline_val is not None, "Offline feature returned None!"
    assert online_val is not None, "Online feature returned None!"
    assert abs(offline_val - online_val) < 1e-5, (
        f"FTI Contract Broken! Offline: {offline_val}, Online: {online_val}"
    )
