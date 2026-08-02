"""
The Continuous Trigger: Compares the baseline feature distribution (from training)
against the recent online feature distribution. If drift is detected, it exits
with a non-zero code to trigger a CI/CD retrain pipeline.
"""

import sys

import pandas as pd
from scipy.stats import ks_2samp

import feast

DRIFT_THRESHOLD = 0.05  # Standard statistical significance


def get_baseline_features():
    """Gets the baseline distribution from the first 6 months of the offline store."""
    fs = feast.FeatureStore(repo_path="feast/")
    entity_df = pd.read_sql(
        """SELECT DISTINCT store_id, event_timestamp 
           FROM f_daily_store_sales 
           WHERE event_timestamp < NOW() - INTERVAL '6 months'""",
        con="postgresql://fti_user:fti_password@localhost:5432/fti_db",
    )
    df = (
        fs.get_historical_features(
            entity_df=entity_df, features=["store_sales_features:rolling_7d_avg_sales"]
        )
        .to_df()
        .dropna()
    )
    return df["rolling_7d_avg_sales"].values


def get_recent_online_features():
    """Gets the most recent feature distribution from the online store (Redis)."""
    fs = feast.FeatureStore(repo_path="feast/")
    # Get the latest state for all 50 stores
    entity_rows = [{"store_id": i} for i in range(1, 51)]
    online_df = (
        fs.get_online_features(
            features=["store_sales_features:rolling_7d_avg_sales"],
            entity_rows=entity_rows,
        )
        .to_df()
        .dropna()
    )
    return online_df["rolling_7d_avg_sales"].values


def check_drift():
    print("Checking for feature drift...")
    baseline = get_baseline_features()
    recent = get_recent_online_features()

    if len(recent) == 0:
        print("❌ No recent online features found. Cannot check drift.")
        sys.exit(1)

    # Run Kolmogorov-Smirnov test
    stat, p_value = ks_2samp(baseline, recent)
    print(f"  KS Statistic: {stat:.4f}, p-value: {p_value:.4f}")

    if p_value < DRIFT_THRESHOLD:
        print(
            f"\n🚨 DRIFT DETECTED! p-value ({p_value:.4f}) < threshold ({DRIFT_THRESHOLD})."
        )
        print("   Triggering retrain pipeline...")
        sys.exit(1)  # Exit with error code to trigger CI/CD
    else:
        print("\n✅ NO DRIFT. Distributions are statistically similar.")
        sys.exit(0)


if __name__ == "__main__":
    check_drift()
