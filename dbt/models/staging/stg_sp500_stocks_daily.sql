{{ config(
    materialized='external',
    location="gs://finance-data-lake-unique-id/silver/sp500_stock_prices.parquet",
    format='parquet'
) }}

WITH source AS (
    SELECT * FROM {{ source('gcs_raw', 'sp500_stock_prices') }}
),

cleaned AS (
    SELECT
        Ticker,
        Date,
        TRY_CAST(Open AS DOUBLE) as Open,
        TRY_CAST(High AS DOUBLE) as High,
        TRY_CAST(Low AS DOUBLE) as Low,
        TRY_CAST(Close AS DOUBLE) as Close,
        TRY_CAST(AdjClose AS DOUBLE) as AdjClose,
        TRY_CAST(Volume AS BIGINT) as Volume
    FROM source
    WHERE Ticker IS NOT NULL 
      AND Date IS NOT NULL
)

SELECT * FROM cleaned
WHERE Close IS NOT NULL AND Volume IS NOT NULL
