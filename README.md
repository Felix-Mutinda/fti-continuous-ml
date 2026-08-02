# FTI Continuous ML: The Decoupling Contract in OSS

This repository implements the **Feature/Training/Inference (FTI)** decoupling contract, pioneered by Hopsworks, using entirely open-source tools. 

It proves that training-serving skew can be prevented *structurally* by enforcing that Feature, Training, and Inference pipelines communicate **only** through a centralized Feature Store.

## The Stack
- **Feature Pipeline:** `dbt` (Batch computation, versioned)
- **Feature Store:** `Feast` (Offline: Postgres, Online: Redis)
- **Training & Registry:** `MLflow` (Point-in-time retrieval, model gating)
- **Drift Trigger:** Custom Python (Population Stability Index)
- **Environment:** `uv` (Fast Python package management)

## Prerequisites
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Docker & Docker Compose (For Postgres, Redis, and MLflow tracking)

## Quickstart

1. **Start Infrastructure:**
   ```bash
   docker-compose up -d
   ```

2. **Sync Python Environment:**
   ```bash
   uv sync
   ```

3. **Run the Full FTI Pipeline:**
   ```bash
   # 1. Ingest data and build features (dbt)
   uv run python data/download_data.py
   uv run dbt run --project-dir dbt
   
   # 2. Materialize to Feature Store (Feast)
   uv run feast -c feast/feature_repo apply
   uv run feast -c feast/feature_repo materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
   
   # 3. Train and Register Model (MLflow)
   uv run python training/train_model.py
   
   # 4. Run the Parity Check (The Proof)
   uv run pytest tests/test_feature_parity.py -v
   ```

## Architecture & Phased Implementation
See the [Implementation Guide](#implementation-guide) below for a step-by-step breakdown of how each layer of the FTI contract is built and verified.

---

### Modular Phased Implementation Plan

#### Phase 1: Infrastructure & Data Ingestion
**Goal:** Stand up the offline/online stores and get raw data into the system.
1. **Docker Compose:** Create `docker-compose.yml` to spin up Postgres (Offline Store + dbt target), Redis (Online Store), and MLflow (Tracking/Registry).
2. **Data Download:** Write `data/download_data.py` to fetch a subset of the Rossmann dataset, clean it slightly, and dump it into the `data/raw/` directory and load it into the Postgres `raw_sales` table.
* **Phase 1 Proof:** You can query the `raw_sales` table in Postgres and see 10,000+ rows.

#### Phase 2: The Feature Pipeline (`dbt`)
**Goal:** Transform raw data into point-in-time-correct features. This is the "F" in FTI.
1. **dbt Setup:** Initialize dbt (`uv run dbt init fti_dbt`), configure `profiles.yml` to point to the local Postgres instance.
2. **Staging Models:** Write `stg_sales.sql` to cast types and handle nulls.
3. **Feature Models:** Write `f_daily_store_sales.sql`. This is the critical step: implement **rolling window functions** (e.g., `AVG(sales) OVER (PARTITION BY store ORDER BY date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING)`). *Crucial: Ensure the window looks strictly backward to prevent data leakage.*
* **Phase 2 Proof:** Run `uv run dbt run`. Query the resulting `f_daily_store_sales` table in Postgres. Verify the rolling averages are mathematically correct and strictly backward-looking.

#### Phase 3: The Feature Store (`Feast`)
**Goal:** Bridge the offline (training) and online (inference) worlds using a single source of truth.
1. **Feast Initialization:** Create `feast/feature_repo/feature_store.yaml` pointing to Postgres (offline) and Redis (online).
2. **Define Entities & Views:** In `entities.py`, define `store_id`. In `features.py`, create a `FeatureView` that maps to the `f_daily_store_sales` dbt table.
3. **Materialization:** Run `feast apply` to register the schema. Run `feast materialize` to push the Postgres data into Redis.
* **Phase 3 Proof:** Use the Feast Python SDK to query both the offline and online stores for `store_id=1` on a specific date. The values must match exactly.

#### Phase 4: Training & Registry (`MLflow`)
**Goal:** Train a model using *only* the feature store, and register it with a promotion gate.
1. **Point-in-Time Retrieval:** Write `training/train_model.py`. Use Feast's `get_historical_features()` to pull the training dataset. *Notice we do not write a single SQL query here; we only ask Feast for features.*
2. **Model Training:** Train a simple XGBoost regressor.
3. **MLflow Logging:** Log parameters, metrics (RMSE), and the model artifact to MLflow. Register the model with the alias `candidate`.
4. **Promotion Gate:** Write `training/promote_model.py`. Compare the `candidate` model against the current `production` model on a holdout set. If it wins, update the MLflow alias to `production`.
* **Phase 4 Proof:** Open the MLflow UI (`http://localhost:5000`). Verify the model is registered, metrics are logged, and the `production` alias is correctly assigned.

#### Phase 5: Drift Detection & Inference
**Goal:** Close the loop. Prove that inference is decoupled from training, and that drift triggers a retrain.
1. **Online Inference:** Write `inference/predict.py`. It must use Feast's `get_online_features()` (hitting Redis) and load the model from MLflow using the `production` alias. It should *never* import dbt or touch Postgres.
2. **Drift Trigger:** Write `training/drift_detector.py`. Calculate the Population Stability Index (PSI) between the baseline training features and the last 24 hours of online features. 
3. **Simulate Drift:** Write `inference/simulate_drift.py` to artificially inject a 20% increase in sales for a specific store in Redis. Run the drift detector.
* **Phase 5 Proof:** The drift detector outputs a warning and exits with a non-zero code when drift is simulated. The inference script successfully returns a prediction using *only* online data.

#### Phase 6: The Payoff (Parity Testing)
**Goal:** Prove the FTI contract is unbroken. This is the ultimate "show, don't tell" artifact.
1. **Write the Test:** Create `tests/test_feature_parity.py`. 
2. **The Logic:** 
   - Pick a random `store_id` and `timestamp`.
   - Query the offline Postgres database directly (bypassing Feast) for the feature value.
   - Query the online Redis database directly (bypassing Feast) for the feature value.
   - Assert they are identical.
3. **CI Integration:** Add this test to your CI pipeline. If a developer changes a dbt SQL file but forgets to update the Feast schema, or if a floating-point precision issue occurs during Redis serialization, this test will fail.
* **Phase 6 Proof:** `uv run pytest tests/test_feature_parity.py -v` passes with a green checkmark. 
