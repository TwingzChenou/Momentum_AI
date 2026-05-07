{{ config(
    materialized='external',
    location="gs://finance-data-lake-unique-id/gold/etf_features.parquet",
    format='parquet'
) }}

WITH base AS (
    SELECT * FROM {{ source('gcs_silver_ext', 'data_raw_etf_weekly') }}
    UNION ALL
    SELECT * FROM {{ source('gcs_silver_ext', 'data_raw_sp500_weekly') }}
),

indicators AS (
    SELECT 
        Ticker,
        Date,
        Close,
        -- SMA basées sur l'optimisation
        AVG(Close) OVER (PARTITION BY Ticker ORDER BY Date ROWS BETWEEN ({{ var('etf_sma_fast') }} - 1) PRECEDING AND CURRENT ROW) as SMA_fast,
        AVG(Close) OVER (PARTITION BY Ticker ORDER BY Date ROWS BETWEEN ({{ var('etf_sma_slow') }} - 1) PRECEDING AND CURRENT ROW) as SMA_slow,
        -- Momentum basé sur l'optimisation (Correction des parenthèses OVER)
        (Close - LAG(Close, {{ var('etf_mom_period') }} ) OVER (PARTITION BY Ticker ORDER BY Date)) / NULLIF(LAG(Close, {{ var('etf_mom_period') }} ) OVER (PARTITION BY Ticker ORDER BY Date), 0) as Momentum_XM
    FROM base
)

SELECT * FROM indicators
WHERE SMA_slow IS NOT NULL
