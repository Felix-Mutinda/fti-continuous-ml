SELECT
    store_id::int AS store_id,
    date::date AS date,
    sales::int AS sales,
    customers::int AS customers,
    open::int AS open,
    promo::int AS promo,
    state_holiday,
    assortment
FROM {{ source('raw', 'raw_sales') }}