{{ config(
    materialized='external',
    location="gs://finance-data-lake-unique-id/silver/data_raw_sp500.parquet",
    format='parquet'
) }}

SELECT * FROM {{ source('gcs_raw', 'data_raw_sp500') }}
WHERE Date IS NOT NULL
