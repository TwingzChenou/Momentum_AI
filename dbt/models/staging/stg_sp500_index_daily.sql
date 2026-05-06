{{ config(
    location="gs://finance-data-lake-unique-id/silver/data_raw_sp500"
) }}

{{ clean_ohlcv(source('gcs_raw', 'data_raw_sp500')) }}
