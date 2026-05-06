{{ config(
    materialized='incremental',
    incremental_strategy='merge',
    unique_key=['Ticker', 'Date'],
    file_format='delta',
    location="gs://finance-data-lake-unique-id/silver/sp500_stock_prices"
) }}

WITH raw_data AS (
    SELECT * FROM {{ source('gcs_raw', 'sp500_stock_prices') }}
)

SELECT
    CAST(Ticker AS STRING) as Ticker,
    CAST(Date AS DATE) as Date,
    CAST(Open AS DOUBLE) as Open,
    CAST(High AS DOUBLE) as High,
    CAST(Low AS DOUBLE) as Low,
    CAST(Close AS DOUBLE) as Close,
    CAST(AdjClose AS DOUBLE) as AdjClose,
    CAST(Volume AS LONG) as Volume
FROM raw_data
WHERE Ticker IS NOT NULL AND Date IS NOT NULL
{% if is_incremental() %}
  AND Date >= (SELECT MAX(Date) FROM {{ this }})
{% endif %}
