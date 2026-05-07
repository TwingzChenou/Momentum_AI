{% macro register_external_sources() %}
    {% set bucket = "finance-data-lake-unique-id" %}
    {% set access_key = env_var('GCS_ACCESS_KEY_ID') %}
    {% set secret_key = env_var('GCS_SECRET_ACCESS_KEY') %}
    
    {% set queries = [
        "INSTALL httpfs",
        "LOAD httpfs",
        "INSTALL delta",
        "LOAD delta",
        "CREATE SECRET IF NOT EXISTS (TYPE GCS, KEY_ID '" ~ access_key ~ "', SECRET '" ~ secret_key ~ "')",
        "SET s3_endpoint='storage.googleapis.com'",
        "CREATE SCHEMA IF NOT EXISTS " ~ target.database ~ ".gcs_raw",
        "CREATE SCHEMA IF NOT EXISTS " ~ target.database ~ ".gcs_silver_ext",
        
        "CREATE OR REPLACE VIEW " ~ target.database ~ ".gcs_raw.data_raw_2b AS SELECT * FROM delta_scan('gs://" ~ bucket ~ "/bronze/data_raw_2b')",
        "CREATE OR REPLACE VIEW " ~ target.database ~ ".gcs_raw.data_raw_etf AS SELECT * FROM delta_scan('gs://" ~ bucket ~ "/bronze/data_raw_etf')",
        "CREATE OR REPLACE VIEW " ~ target.database ~ ".gcs_raw.data_raw_sp500 AS SELECT * FROM delta_scan('gs://" ~ bucket ~ "/bronze/data_raw_sp500')",
        "CREATE OR REPLACE VIEW " ~ target.database ~ ".gcs_raw.sp500_stock_prices AS SELECT * FROM delta_scan('gs://" ~ bucket ~ "/bronze/sp500_stock_prices')",
        
        "CREATE OR REPLACE VIEW " ~ target.database ~ ".gcs_silver_ext.data_raw_2b_weekly AS SELECT * FROM delta_scan('gs://" ~ bucket ~ "/silver/data_raw_2b_weekly')",
        "CREATE OR REPLACE VIEW " ~ target.database ~ ".gcs_silver_ext.data_raw_etf_weekly AS SELECT * FROM delta_scan('gs://" ~ bucket ~ "/silver/data_raw_etf_weekly')",
        "CREATE OR REPLACE VIEW " ~ target.database ~ ".gcs_silver_ext.data_raw_sp500_weekly AS SELECT * FROM delta_scan('gs://" ~ bucket ~ "/silver/data_raw_sp500_weekly')",
        "CREATE OR REPLACE VIEW " ~ target.database ~ ".gcs_silver_ext.sp500_stock_prices_weekly AS SELECT * FROM delta_scan('gs://" ~ bucket ~ "/silver/sp500_stock_prices_weekly')"
    ] %}

    {% for query in queries %}
        {% do run_query(query) %}
    {% endfor %}
    
    {{ log("✅ Sources externes GCS enregistrées dans DuckDB", info=True) }}
{% endmacro %}
