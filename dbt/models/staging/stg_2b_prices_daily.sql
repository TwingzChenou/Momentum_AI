{{ config(
    materialized='external',
    location="gs://finance-data-lake-unique-id/silver/data_raw_2b.parquet",
    format='parquet'
) }}

SELECT * FROM {{ source('gcs_raw', 'data_raw_2b') }}
WHERE Ticker IS NOT NULL AND Date IS NOT NULL
