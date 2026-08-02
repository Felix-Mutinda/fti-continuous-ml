# scripts/feast_materialize.py
import os
import subprocess
import sys
from datetime import datetime, timedelta

import pytz


def run_feast_pipeline():
    # Locate the feast executable in the current uv virtual environment
    venv_bin = os.path.dirname(sys.executable)
    feast_cmd = os.path.join(venv_bin, "feast.exe" if os.name == "nt" else "feast")

    print("Applying feature definitions to registry via Feast CLI...")
    subprocess.run([feast_cmd, "-c", "feast/", "apply"], check=True)

    print("Materializing data from Offline (Postgres) to Online (Redis)...")
    end_date = datetime.now(pytz.utc)
    start_date = end_date - timedelta(days=800)  # Cover our 730 days of data

    # Format dates for the CLI
    start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    subprocess.run(
        [feast_cmd, "-c", "feast/", "materialize", start_str, end_str], check=True
    )

    print("Materialization complete!")


if __name__ == "__main__":
    run_feast_pipeline()
