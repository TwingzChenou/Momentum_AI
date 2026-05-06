{% macro drop_old_gold_tables() %}
    {% set tables = ['gold_stock_features', 'gold_etf_features', 'gold_sp500_index_features'] %}
    {% set schemas = ['default', 'gold_layer'] %}
    {% for schema in schemas %}
        {% for table in tables %}
            {% set drop_query %}
                DROP TABLE IF EXISTS {{ schema }}.{{ table }}
            {% endset %}
            {{ log("Dropping table " ~ schema ~ "." ~ table, info=True) }}
            {% do run_query(drop_query) %}
        {% endfor %}
    {% endfor %}
{% endmacro %}
