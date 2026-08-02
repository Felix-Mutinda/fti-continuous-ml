SELECT
    store_id,
    date,
    sales,
    
    -- Rolling 7-day average (strictly backward looking to prevent leakage)
    AVG(sales) OVER (
        PARTITION BY store_id 
        ORDER BY date 
        ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    )::float AS rolling_7d_avg_sales,
    
    -- Rolling 30-day average
    AVG(sales) OVER (
        PARTITION BY store_id 
        ORDER BY date 
        ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
    )::float AS rolling_30d_avg_sales,
    
    -- Lagged sales (yesterday's sales)
    LAG(sales, 1) OVER (
        PARTITION BY store_id 
        ORDER BY date
    )::int AS lag_1d_sales

FROM {{ ref('stg_sales') }}