{% macro clean_ohlcv(source_table) %}

WITH source AS (
    SELECT * FROM {{ source_table }}
),
standardized AS (
    SELECT
        Ticker,
        Date,
        Open,
        High,
        Low,
        COALESCE(AdjClose, Close) as Close,
        Volume
    FROM source
),
deduplicated AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (PARTITION BY Ticker, Date ORDER BY Date DESC) as rn
    FROM standardized
),
filtered AS (
    SELECT *
    FROM deduplicated
    WHERE rn = 1
    AND (
        -- Soit tout est nul (Master Index)
        (Open IS NULL AND High IS NULL AND Low IS NULL AND Close IS NULL AND Volume IS NULL)
        -- Soit tout est valide mathématiquement
        OR (
            Open > 0 AND High > 0 AND Low > 0 AND Close > 0 AND Volume >= 0
            AND High >= Low
            AND High >= Open
            AND High >= Close
            AND Low <= Open
            AND Low <= Close
        )
    )
)
SELECT
    Ticker,
    Date,
    Open,
    High,
    Low,
    Close,
    Volume
FROM filtered

{% endmacro %}
