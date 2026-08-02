from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float64, Int64

from feast import FeatureView, Field

from .entities import store_id

# Define the source (the dbt output table)
store_sales_source = PostgreSQLSource(
    name="store_sales_source",
    query="SELECT * FROM f_daily_store_sales",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",  # Using a distinct column for the created timestamp
)

# Define the Feature View
store_sales_fv = FeatureView(
    name="store_sales_features",
    entities=[store_id],
    ttl=None,  # We want the latest state, no expiration
    schema=[
        Field(name="sales", dtype=Int64),
        Field(name="rolling_7d_avg_sales", dtype=Float64),
        Field(name="rolling_30d_avg_sales", dtype=Float64),
        Field(name="lag_1d_sales", dtype=Int64),
    ],
    source=store_sales_source,
    online=True,
    tags={"pipeline": "dbt", "phase": "fti"},
)
