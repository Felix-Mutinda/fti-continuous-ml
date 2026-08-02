"""
FTI Promotion Gate: A candidate model ONLY becomes 'production' if it
beats the current production model on a held-out recent window.
This prevents 'continuous chaos' from 'continuous training'.
"""

import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import root_mean_squared_error

import feast

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "fti-demand-forecast"
IMPROVEMENT_THRESHOLD = 0.01  # Candidate must beat production by at least 1%


def get_holdout_data():
    """Pulls the most recent 30 days as a holdout evaluation window."""
    fs = feast.FeatureStore(repo_path="feast/")

    entity_df = pd.read_sql(
        """SELECT DISTINCT store_id, event_timestamp 
           FROM f_daily_store_sales 
           WHERE rolling_7d_avg_sales IS NOT NULL
           AND event_timestamp >= NOW() - INTERVAL '30 days'""",
        con="postgresql://fti_user:fti_password@localhost:5432/fti_db",
    )

    if entity_df.empty:
        # Fallback: use last 30 days of available data
        entity_df = pd.read_sql(
            """SELECT DISTINCT store_id, event_timestamp 
               FROM f_daily_store_sales 
               WHERE rolling_7d_avg_sales IS NOT NULL
               ORDER BY event_timestamp DESC
               LIMIT 1500""",
            con="postgresql://fti_user:fti_password@localhost:5432/fti_db",
        )

    df = (
        fs.get_historical_features(
            entity_df=entity_df,
            features=[
                "store_sales_features:sales",
                "store_sales_features:rolling_7d_avg_sales",
                "store_sales_features:rolling_30d_avg_sales",
                "store_sales_features:lag_1d_sales",
            ],
        )
        .to_df()
        .dropna()
    )

    return df


def evaluate_model(model, df):
    """Evaluates a model on the holdout set."""
    feature_cols = ["rolling_7d_avg_sales", "rolling_30d_avg_sales", "lag_1d_sales"]
    X = df[feature_cols]
    y = df["sales"]
    y_pred = model.predict(X)
    return root_mean_squared_error(y, y_pred)


def promote():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    print("Evaluating promotion gate...")

    # Get holdout data
    holdout_df = get_holdout_data()
    if holdout_df.empty:
        print("❌ No holdout data available. Aborting promotion.")
        return

    # Load candidate model
    try:
        candidate_uri = f"models:/{MODEL_NAME}@candidate"
        candidate_model = mlflow.xgboost.load_model(candidate_uri)
        candidate_rmse = evaluate_model(candidate_model, holdout_df)
        print(f"  Candidate RMSE on holdout: {candidate_rmse:.2f}")
    except Exception as e:
        print(f"❌ No candidate model found: {e}")
        return

    # Load current production model (if exists)
    try:
        production_uri = f"models:/{MODEL_NAME}@production"
        production_model = mlflow.xgboost.load_model(production_uri)
        production_rmse = evaluate_model(production_model, holdout_df)
        print(f"  Production RMSE on holdout: {production_rmse:.2f}")
    except Exception:
        print("  No existing production model. Promoting candidate directly.")
        production_rmse = float("inf")

    # Promotion decision
    improvement = (
        (production_rmse - candidate_rmse) / production_rmse
        if production_rmse != float("inf")
        else 1.0
    )

    if improvement >= IMPROVEMENT_THRESHOLD or production_rmse == float("inf"):
        # Get candidate version number
        candidate_version = client.get_model_version_by_alias(
            MODEL_NAME, "candidate"
        ).version

        # Promote: set 'production' alias to candidate
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="production",
            version=candidate_version,
        )
        print(f"\n✅ PROMOTED: Version {candidate_version} is now 'production'.")
        print(
            f"   Improvement: {improvement * 100:.1f}% (threshold: {IMPROVEMENT_THRESHOLD * 100:.1f}%)"
        )
    else:
        print(
            f"\n❌ REJECTED: Candidate only improved by {improvement * 100:.1f}% (need {IMPROVEMENT_THRESHOLD * 100:.1f}%)."
        )
        print("   Current production model remains active.")


if __name__ == "__main__":
    promote()
