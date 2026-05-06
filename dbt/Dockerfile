FROM python:3.10-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install dbt-duckdb and dependencies
RUN pip install --no-cache-dir \
    dbt-duckdb \
    duckdb \
    boto3 \
    pandas

# Set working directory
WORKDIR /app/dbt

# Copy dbt project (excluding artifacts already deleted)
COPY dbt/ /app/dbt/

# Set dbt profiles directory
ENV DBT_PROFILES_DIR=/app/dbt

# Script to run dbt
COPY <<EOF /usr/local/bin/run-dbt.sh
#!/bin/bash
dbt "$@"
EOF

RUN chmod +x /usr/local/bin/run-dbt.sh

ENTRYPOINT ["/usr/local/bin/run-dbt.sh"]
