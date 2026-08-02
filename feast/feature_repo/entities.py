from feast import Entity, ValueType

store_id = Entity(
    name="store_id",
    value_type=ValueType.INT64,
    description="Unique identifier for the retail store",
)
