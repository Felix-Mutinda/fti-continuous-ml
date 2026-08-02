"""
THE 'I' IN FTI: Inference Pipeline.
Completely decoupled from training. Reads ONLY from the Online Store (Redis)
and loads ONLY the 'production' model from MLflow.
"""

import mlflow
import mlflow.xgboost

import feast

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "fti-demand-forecast"


def predict_daily_demand(store_id: int) -> float:
    # 1. Pull CURRENT features from the Online Store (Redis) via Feast
    fs = feast.FeatureStore(repo_path="feast/")
    online_features = fs.get_online_features(
        features=[
            "store_sales_features:rolling_7d_avg_sales",
            "store_sales_features:rolling_30d_avg_sales",
            "store_sales_features:lag_1d_sales",
        ],
        entity_rows=[{"store_id": store_id}],
    ).to_df()

    # Check for nulls (e.g., if a store is brand new and hasn't been materialized)
    if online_features.isnull().any().any():
        raise ValueError(
            f"Missing online features for store {store_id}. Has it been materialized?"
        )

    # 2. Load the 'production' model from MLflow Registry
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}@production"
    model = mlflow.xgboost.load_model(model_uri)

    # 3. Predict
    feature_cols = ["rolling_7d_avg_sales", "rolling_30d_avg_sales", "lag_1d_sales"]
    X = online_features[feature_cols]
    prediction = model.predict(X)[0]

    return prediction


if __name__ == "__main__":
    store_to_predict = 1
    forecast = predict_daily_demand(store_to_predict)
    print(f"🔮 Predicted demand for Store {store_to_predict}: {forecast:.2f} units")
