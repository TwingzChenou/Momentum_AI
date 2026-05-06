{{ config(
    location="gs://finance-data-lake-unique-id/silver/data_raw_etf"
) }}

{{ clean_ohlcv(source('gcs_raw', 'data_raw_etf')) }}
