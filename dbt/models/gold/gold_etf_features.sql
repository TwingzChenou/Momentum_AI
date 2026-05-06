{% set target_path = var('etf_features_path', 'gs://finance-data-lake-unique-id/gold/etf_features') %}
{{ config(
    materialized='table',
    file_format='delta',
    location=target_path
) }}
{{ log("Writing gold_etf_features to: " ~ target_path, info=True) }}

WITH base AS (
    SELECT Ticker, Date, Close FROM {{ source('gcs_silver_ext', 'data_raw_etf_weekly') }}
    UNION ALL
    -- On ajoute l'indice S&P 500 (avec son ticker de référence)
    SELECT '^GSPC' as Ticker, Date, Close FROM {{ source('gcs_silver_ext', 'data_raw_sp500_weekly') }}
),

indicators AS (
    SELECT 
        Ticker,
        Date,
        Close,
        -- SMA 50
        AVG(Close) OVER (PARTITION BY Ticker ORDER BY Date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) as SMA_fast,
        -- SMA 200
        AVG(Close) OVER (PARTITION BY Ticker ORDER BY Date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) as SMA_slow,
        -- Momentum (Variation sur 1 période pour les benchmarks)
        (Close - LAG(Close, 1) OVER (PARTITION BY Ticker ORDER BY Date)) / LAG(Close, 1) OVER (PARTITION BY Ticker ORDER BY Date) as Momentum_XM
    FROM base
)

SELECT * FROM indicators
WHERE SMA_slow IS NOT NULL
