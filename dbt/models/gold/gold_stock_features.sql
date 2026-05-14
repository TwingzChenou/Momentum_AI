{{ config(
    materialized='external',
    location="gs://finance-data-lake-unique-id/gold/stock_features.parquet",
    format='parquet'
) }}

WITH base AS (
    SELECT * FROM {{ source('gcs_silver_ext', 'data_raw_2b_weekly') }}
),

-- Préparation des valeurs précédentes
pre_indicators AS (
    SELECT 
        *,
        LAG(Close) OVER (PARTITION BY Ticker ORDER BY Date) as prev_close,
        LAG(High) OVER (PARTITION BY Ticker ORDER BY Date) as prev_high,
        LAG(Low) OVER (PARTITION BY Ticker ORDER BY Date) as prev_low
    FROM base
),

-- Calcul du True Range (TR) et des Directional Movements (DM)
true_range_calc AS (
    SELECT 
        *,
        GREATEST(
            High - Low, 
            ABS(High - prev_close), 
            ABS(Low - prev_close)
        ) as TR,
        CASE WHEN (High - prev_high) > (prev_low - Low) AND (High - prev_high) > 0 THEN (High - prev_high) ELSE 0 END as DM_plus,
        CASE WHEN (prev_low - Low) > (High - prev_high) AND (prev_low - Low) > 0 THEN (prev_low - Low) ELSE 0 END as DM_minus
    FROM pre_indicators
),

-- Lissage des indicateurs (Moyennes mobiles basées sur l'optimisation)
smoothed_indicators AS (
    SELECT 
        *,
        AVG(TR) OVER (PARTITION BY Ticker ORDER BY Date ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) as ATR,
        AVG(DM_plus) OVER (PARTITION BY Ticker ORDER BY Date ROWS BETWEEN 8 PRECEDING AND CURRENT ROW) as DM_plus_smooth,
        AVG(DM_minus) OVER (PARTITION BY Ticker ORDER BY Date ROWS BETWEEN 8 PRECEDING AND CURRENT ROW) as DM_minus_smooth,
        AVG(Close) OVER (PARTITION BY Ticker ORDER BY Date ROWS BETWEEN ({{ var('stock_sma_fast') }} - 1) PRECEDING AND CURRENT ROW) as SMA_fast,
        AVG(Close) OVER (PARTITION BY Ticker ORDER BY Date ROWS BETWEEN ({{ var('stock_sma_slow') }} - 1) PRECEDING AND CURRENT ROW) as SMA_slow,
        (Close - LAG(Close, {{ var('stock_mom_period') }} ) OVER (PARTITION BY Ticker ORDER BY Date)) / LAG(Close, {{ var('stock_mom_period') }} ) OVER (PARTITION BY Ticker ORDER BY Date) as Momentum_XM
    FROM true_range_calc
),

-- Calcul des Directional Indicators (DI)
final_adx_calc AS (
    SELECT 
        *,
        100 * (DM_plus_smooth / NULLIF(ATR, 0)) as DI_plus,
        100 * (DM_minus_smooth / NULLIF(ATR, 0)) as DI_minus
    FROM smoothed_indicators
),

-- Assemblage final
final_features AS (
    SELECT 
        Ticker, Date, Close, SMA_fast, SMA_slow, Momentum_XM, ATR,
        -- Calcul explicite en pourcentage (ex: 15.0 pour 15%)
        (ATR / NULLIF(Close, 0)) * 100 as ATR_pct,
        AVG(100 * ABS(DI_plus - DI_minus) / NULLIF(DI_plus + DI_minus, 0)) 
            OVER (PARTITION BY Ticker ORDER BY Date ROWS BETWEEN 8 PRECEDING AND CURRENT ROW) as ADX
    FROM final_adx_calc
)

SELECT * FROM final_features
WHERE SMA_slow IS NOT NULL AND ADX IS NOT NULL
