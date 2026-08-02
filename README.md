# FTI Continuous ML: The Decoupling Contract in OSS

This repository implements the **Feature/Training/Inference (FTI)** decoupling contract, pioneered by Hopsworks and detailed in Jim Dowling’s O’Reilly book *[Building Machine Learning Systems with a Feature Store](https://www.hopsworks.ai/lp/full-book-oreilly-building-machine-learning-systems-with-a-feature-store)*, using entirely open-source tools.

It proves that training-serving skew can be prevented *structurally* by enforcing that Feature, Training, and Inference pipelines communicate **only** through a centralized Feature Store. 

## The Philosophy
In the FTI paradigm, feature pipelines, training pipelines, and inference pipelines are three entirely independent artifacts. Training never touches raw data; inference never recomputes a feature from scratch. By building this contract from scratch using open-source tools, we prove the concept is tool-agnostic, deeply understood, and entirely portable.

## The Stack
- **Environment:** `uv` (Blazing-fast Python package management)
- **Feature Pipeline (F):** `dbt` (Batch computation, versioned, strictly backward-looking windows)
- **Feature Store:** `Feast` (Offline: Postgres, Online: Redis)
- **Training & Registry (T):** `MLflow` (Point-in-time retrieval, gated model promotion)
- **Drift Trigger:** Custom Python using `scipy` (Kolmogorov-Smirnov statistical tests)
- **Verification:** `pytest` (Automated parity and architectural decoupling guards)

## Prerequisites
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Docker & Docker Compose (For Postgres, Redis)

## Quickstart

### 1. Start Infrastructure & Sync Environment
```bash
docker-compose up -d
uv sync
```

### 2. Run the Full FTI Pipeline
*(Note: We use Python wrapper scripts for Feast/dbt operations to ensure cross-platform compatibility, avoiding bash-specific commands).*

```bash
# 1. Ingest synthetic data (mirroring Rossmann schema) and build features (dbt)
uv run python data/ingest_data.py
uv run dbt run --project-dir dbt --profiles-dir dbt

# 2. Materialize to Feature Store (Feast)
uv run python scripts/feast_materialize.py

# 3. Train and Register Model (MLflow)
# (Ensure MLflow UI is running in another tab: uv run mlflow ui --host 127.0.0.1 --port 5000)
uv run python training/train_model.py
uv run python training/promote_model.py

# 4. Run the Parity & Architecture Checks (The Proof)
uv run pytest tests/ -v
```

## The "Show, Don't Tell" Proof (Automated Testing)
We don't measure success by model accuracy; we measure the health of the FTI contract itself. The `tests/` directory contains automated guards that run in CI/CD:

1. **`test_feature_parity.py`**: The ultimate FTI proof. It queries Postgres (offline) and Redis (online) directly for the exact same entity/timestamp and asserts they are mathematically identical down to the floating point.
2. **`test_architecture.py`**: Uses Python’s `ast` module to parse `inference/predict.py` and assert that it *never* imports raw database drivers (`sqlalchemy`, `psycopg2`) or `dbt`. If a developer accidentally couples inference to the database, the build fails.
3. **`test_drift.py`**: Validates the underlying statistical logic of our custom KS-test drift detector.

## CI/CD: The Continuous Loop
The `.github/workflows/continuous-ml.yml` file doesn't just run tests; it **executes the entire FTI lifecycle** on every push. It spins up the infrastructure, trains a baseline model, simulates concept drift, verifies the drift trigger catches it, retrains the model, and re-verifies parity. It proves the pipeline can heal itself automatically.

---

## Modular Phased Implementation Guide

### Phase 1: Infrastructure & Data Ingestion
- **Goal:** Stand up the offline/online stores and get raw data into the system.
- **Action:** `docker-compose.yml` spins up Postgres and Redis. `data/ingest_data.py` generates a highly realistic synthetic dataset (mirroring the Rossmann schema to avoid Kaggle API friction) and loads it into Postgres.

### Phase 2: The Feature Pipeline (`dbt`)
- **Goal:** Transform raw data into point-in-time-correct features.
- **Action:** Implement strict, backward-looking rolling window functions in `f_daily_store_sales.sql` (e.g., `ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING`) to guarantee zero data leakage. Materialized as a physical table for Feast compatibility.

### Phase 3: The Feature Store (`Feast`)
- **Goal:** Bridge the offline and online worlds using a single source of truth.
- **Action:** Map the dbt output to a Feast `FeatureView`. `scripts/feast_materialize.py` pushes the Postgres data into Redis. 

### Phase 4: Training & Registry (`MLflow`)
- **Goal:** Train a model using *only* the feature store, and register it with a promotion gate.
- **Action:** `training/train_model.py` uses Feast's `get_historical_features()` (no raw SQL). `training/promote_model.py` compares the `candidate` model against the `production` model on a holdout set before updating the MLflow alias.

### Phase 5: Drift Detection & Inference
- **Goal:** Close the loop. Prove inference is decoupled, and drift triggers a retrain.
- **Action:** `inference/predict.py` reads *only* from Redis and MLflow. `training/drift_detector.py` calculates the KS-test statistic between baseline and recent online features. `inference/simulate_drift.py` artificially inflates sales to prove the trigger works.

### Phase 6: The Payoff (Parity Testing)
- **Goal:** Prove the FTI contract is unbroken.
- **Action:** Write the `pytest` suite that mathematically and structurally proves the system works, ensuring it can never silently degrade.