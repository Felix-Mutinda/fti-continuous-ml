"""
Simulates real-world concept drift by inflating recent sales data,
then re-materializes to the online store to trigger the drift detector.
"""

import os
import subprocess
import sys
from datetime import datetime, timedelta

import redis
from sqlalchemy import create_engine, text


def inject_drift():
    print("Injecting concept drift into the offline store...")
    engine = create_engine("postgresql://fti_user:fti_password@localhost:5432/fti_db")

    # Inflate sales by 50% for the last 10 days across all stores
    # This will structurally shift the rolling 7-day and 30-day averages
    cutoff_date = (datetime.now() - timedelta(days=10)).date()

    with engine.connect() as conn:
        # Update raw sales
        conn.execute(
            text("""
            UPDATE raw_sales 
            SET sales = CAST(sales * 1.5 AS INTEGER),
                customers = CAST(customers * 1.5 AS INTEGER)
            WHERE date >= :cutoff
        """),
            {"cutoff": cutoff_date},
        )
        conn.commit()
    print(f"  Inflated sales by 50% for dates >= {cutoff_date}.")

    # Locate the dbt executable in the current uv virtual environment
    venv_bin = os.path.dirname(sys.executable)
    dbt_cmd = os.path.join(venv_bin, "dbt.exe" if os.name == "nt" else "dbt")

    # Re-run dbt to recalculate the rolling features with the new inflated data
    print("Re-running dbt to propagate drift to feature tables...")
    subprocess.run(
        [dbt_cmd, "run", "--project-dir", "dbt", "--profiles-dir", "dbt"], check=True
    )

    # FLUSH REDIS to guarantee a clean slate for the online store
    print("Flushing Redis to guarantee a clean online store...")
    r = redis.Redis(host="localhost", port=6379, db=0)
    r.flushdb()

    # Re-materialize to push the drifted features to Redis
    print("Re-materializing drifted features to the online store...")
    feast_cmd = os.path.join(venv_bin, "feast.exe" if os.name == "nt" else "feast")

    end_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    start_date = (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%dT%H:%M:%SZ")

    subprocess.run(
        [feast_cmd, "-c", "feast/", "materialize", start_date, end_date], check=True
    )
    print("✅ Drift successfully injected and materialized to Redis.")


if __name__ == "__main__":
    inject_drift()
