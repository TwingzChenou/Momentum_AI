"""
Ce module centralise toutes les requêtes Spark-SQL utilisées dans le pipeline de données Momentum AI.
"""

# Requête pour le calcul des indicateurs techniques (Gold Layer)
# Utilise des placeholders {variable} pour injecter les paramètres de la stratégie
QUERY_COMPUTE_GOLD_INDICATORS = """
WITH base AS (
    SELECT 
        Ticker,
        Date,
        Open,
        High,
        Low,
        Close,
        Volume,
        LAG(Close, 1) OVER (PARTITION BY Ticker ORDER BY Date) as prev_close,
        LAG(High, 1) OVER (PARTITION BY Ticker ORDER BY Date) as prev_high,
        LAG(Low, 1) OVER (PARTITION BY Ticker ORDER BY Date) as prev_low
    FROM temp_silver_data
),

metrics AS (
    SELECT 
        *,
        -- 1. SMA
        AVG(Close) OVER (PARTITION BY Ticker ORDER BY Date ROWS BETWEEN {sma_fast_p} PRECEDING AND CURRENT ROW) as SMA_fast,
        AVG(Close) OVER (PARTITION BY Ticker ORDER BY Date ROWS BETWEEN {sma_slow_p} PRECEDING AND CURRENT ROW) as SMA_slow,
        
        -- 2. Momentum
        (Close - LAG(Close, {mom_p}) OVER (PARTITION BY Ticker ORDER BY Date)) / NULLIF(LAG(Close, {mom_p}) OVER (PARTITION BY Ticker ORDER BY Date), 0) as Momentum_XM,
        
        -- 3. ATR (Average True Range)
        GREATEST(
            High - Low,
            ABS(High - prev_close),
            ABS(Low - prev_close)
        ) as TR
    FROM base
),

atr_calc AS (
    SELECT 
        *,
        AVG(TR) OVER (PARTITION BY Ticker ORDER BY Date ROWS BETWEEN {atr_p} PRECEDING AND CURRENT ROW) as ATR
    FROM metrics
),

adx_base AS (
    SELECT 
        *,
        (ATR / NULLIF(Close, 0)) * 100 as ATR_pct,
        High - prev_high as up_move,
        prev_low - Low as down_move
    FROM atr_calc
),

dm_calc AS (
    SELECT 
        *,
        CASE WHEN up_move > down_move AND up_move > 0 THEN up_move ELSE 0 END as dm_plus,
        CASE WHEN down_move > up_move AND down_move > 0 THEN down_move ELSE 0 END as dm_minus
    FROM adx_base
),

adx_final AS (
    SELECT 
        *,
        AVG(dm_plus) OVER (PARTITION BY Ticker ORDER BY Date ROWS BETWEEN {adx_p} PRECEDING AND CURRENT ROW) as dm_plus_smooth,
        AVG(dm_minus) OVER (PARTITION BY Ticker ORDER BY Date ROWS BETWEEN {adx_p} PRECEDING AND CURRENT ROW) as dm_minus_smooth
    FROM dm_calc
),

di_calc AS (
    SELECT 
        *,
        100 * dm_plus_smooth / NULLIF(ATR, 0) as di_plus,
        100 * dm_minus_smooth / NULLIF(ATR, 0) as di_minus
    FROM adx_final
),

adx_result AS (
    SELECT 
        *,
        100 * ABS(di_plus - di_minus) / NULLIF(di_plus + di_minus, 0) as dx
    FROM di_calc
)

SELECT 
    Ticker,
    Date,
    Open,
    High,
    Low,
    Close,
    Volume,
    SMA_fast,
    SMA_slow,
    Momentum_XM,
    ATR_pct,
    AVG(dx) OVER (PARTITION BY Ticker ORDER BY Date ROWS BETWEEN {adx_p} PRECEDING AND CURRENT ROW) as ADX
FROM adx_result
"""

# Requête pour le rééchantillonnage (Silver Layer)
QUERY_RESAMPLE_WEEKLY = """
SELECT 
    Ticker,
    date_trunc('week', Date) + INTERVAL 4 DAYS as Date, -- Calcule le Vendredi de la semaine
    CAST(FIRST(Open, true) AS DOUBLE) as Open,
    CAST(MAX(High) AS DOUBLE) as High,
    CAST(MIN(Low) AS DOUBLE) as Low,
    CAST(LAST(Close, true) AS DOUBLE) as Close,
    CAST(SUM(Volume) AS DOUBLE) as Volume
FROM bronze_data
WHERE Close IS NOT NULL AND NOT isnan(Close)
GROUP BY Ticker, date_trunc('week', Date)
ORDER BY Ticker, Date
"""
