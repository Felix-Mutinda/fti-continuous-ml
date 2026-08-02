"""
THE 'T' IN FTI: Training Pipeline.
Pulls point-in-time-correct features ONLY from Feast's offline store.
Never touches raw data or dbt directly.
"""

from datetime import datetime

import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

import feast

# Ensure MLflow tracking is local
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "fti-demand-forecast"


def get_training_data_from_feast():
    """
    Pulls point-in-time-correct features from Feast's offline store (Postgres).
    This is the FTI contract: training NEVER queries raw tables directly.
    """
    print("Pulling point-in-time features from Feast Offline Store...")
    fs = feast.FeatureStore(repo_path="feast/")

    # Define the entity dataframe: all stores, all dates
    # Feast will join features correctly based on event_timestamp
    entity_df = pd.read_sql(
        "SELECT DISTINCT store_id, event_timestamp FROM f_daily_store_sales WHERE rolling_7d_avg_sales IS NOT NULL",
        con="postgresql://fti_user:fti_password@localhost:5432/fti_db",
    )

    # Get historical features (point-in-time correct join)
    training_df = fs.get_historical_features(
        entity_df=entity_df,
        features=[
            "store_sales_features:sales",
            "store_sales_features:rolling_7d_avg_sales",
            "store_sales_features:rolling_30d_avg_sales",
            "store_sales_features:lag_1d_sales",
        ],
    ).to_df()

    # Drop rows with nulls (first 30 days per store won't have full rolling windows)
    training_df = training_df.dropna().reset_index(drop=True)
    print(f"Retrieved {len(training_df)} training samples from Feast.")
    return training_df


def train_and_register(df):
    """Trains XGBoost, logs to MLflow, registers as 'candidate'."""

    # Define features and target
    feature_cols = ["rolling_7d_avg_sales", "rolling_30d_avg_sales", "lag_1d_sales"]
    target_col = "sales"

    X = df[feature_cols]
    y = df[target_col]

    # Time-based split (last 60 days as test set to respect temporal order)
    df_sorted = df.sort_values("event_timestamp").reset_index(drop=True)
    split_idx = len(df_sorted) - (50 * 60)  # 50 stores * 60 days
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # Train
    model = XGBRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0
    )

    # MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("fti-continuous-training")

    with mlflow.start_run(run_name=f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        # Log parameters
        mlflow.log_params(
            {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "feature_cols": str(feature_cols),
                "training_samples": len(X_train),
            }
        )

        # Fit
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        mlflow.log_metrics({"rmse": rmse, "mae": mae})
        print(f"Model trained. RMSE: {rmse:.2f}, MAE: {mae:.2f}")

        # Log and register model
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        # Tag as 'candidate' (NOT production yet)
        client = mlflow.MlflowClient()
        model_version = mlflow.search_registered_models(
            filter_string=f"name='{MODEL_NAME}'"
        )

        # Get the latest version and set alias
        latest = client.search_model_versions(f"name='{MODEL_NAME}'")[0]
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="candidate",
            version=latest.version,
        )
        print(
            f"Model registered as '{MODEL_NAME}' version {latest.version} with alias 'candidate'."
        )

        # Save test metrics for the promotion gate
        mlflow.log_metric("holdout_rmse", rmse)

    return rmse


if __name__ == "__main__":
    df = get_training_data_from_feast()
    rmse = train_and_register(df)
    print(f"\n✅ Training complete. Candidate model RMSE: {rmse:.2f}")
    print("Run 'python training/promote_model.py' to evaluate promotion to Production.")
