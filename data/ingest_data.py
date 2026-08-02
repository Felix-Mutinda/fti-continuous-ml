import time

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


def generate_retail_data(num_stores=50, days=730):
    """
    Generates a synthetic dataset mirroring the Rossmann Store Sales schema.
    Includes built-in seasonality and a slight upward trend to simulate real-world drift.
    """
    print("Generating synthetic retail data...")
    np.random.seed(42)

    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    stores = np.arange(1, num_stores + 1)

    # Create a multi-index for all store/date combinations
    index = pd.MultiIndex.from_product([stores, dates], names=["store_id", "date"])
    df = pd.DataFrame(index=index).reset_index()

    # Base sales with store-specific seasonality and trend
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # Base sales: higher in larger stores, closed on Sundays (day_of_week == 6)
    base_sales = 5000 + (df["store_id"] * 50)
    sunday_closure = (df["day_of_week"] == 6).astype(int)

    # Seasonality (higher in December)
    seasonality = 1000 * np.sin(2 * np.pi * df["month"] / 12)

    # Map the trend to the actual dates in the dataframe to match the 36,500 row length
    trend_map = pd.Series(np.linspace(0, 500, days), index=dates)
    trend = df["date"].map(trend_map)

    # Calculate final sales (0 if closed on Sunday)
    noise = np.random.normal(0, 300, len(df))
    df["sales"] = (base_sales + seasonality + trend + noise) * (1 - sunday_closure)
    df["sales"] = df["sales"].clip(lower=0).astype(int)

    # Customers correlate with sales
    df["customers"] = (
        (df["sales"] / 50 + np.random.normal(0, 20, len(df))).clip(lower=0).astype(int)
    )
    df.loc[df["sales"] == 0, "customers"] = 0

    # Categorical features
    df["open"] = (df["sales"] > 0).astype(int)
    df["promo"] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])
    df["state_holiday"] = np.random.choice(
        ["0", "a", "b", "c"], size=len(df), p=[0.9, 0.05, 0.03, 0.02]
    )
    df["assortment"] = np.random.choice(
        ["a", "b", "c"], size=len(df), p=[0.5, 0.3, 0.2]
    )

    # Select and order columns to match Rossmann schema
    df = df[
        [
            "store_id",
            "date",
            "sales",
            "customers",
            "open",
            "promo",
            "state_holiday",
            "assortment",
        ]
    ]

    # Ensure date is just the date part (no time)
    df["date"] = df["date"].dt.date
    return df


def load_to_postgres(df):
    """Loads the dataframe into the Postgres raw_sales table."""
    print("Connecting to Postgres...")
    engine = create_engine("postgresql://fti_user:fti_password@localhost:5432/fti_db")

    # Wait for Postgres to be fully ready
    for _ in range(10):
        try:
            with engine.connect() as conn:
                # Use sqlalchemy.text for SQLAlchemy 2.0 compatibility
                conn.execute(text("SELECT 1"))
            break
        except Exception:
            print("Waiting for Postgres to be ready...")
            time.sleep(2)

    print("Loading data into raw_sales table...")
    # If table exists, replace it for a clean slate
    df.to_sql("raw_sales", engine, if_exists="replace", index=False)
    print(f"Successfully loaded {len(df)} rows into Postgres.")


if __name__ == "__main__":
    data = generate_retail_data()

    # Save locally as well for reference
    data.to_csv("data/raw/retail_sales.csv", index=False)
    print("Saved raw data to data/raw/retail_sales.csv")

    load_to_postgres(data)
