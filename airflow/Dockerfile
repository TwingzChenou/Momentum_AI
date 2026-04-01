FROM apache/airflow:2.10.2-python3.10

USER root

# Installer Java 17 et procps (Nécessaire pour Spark)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    procps \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configurer JAVA_HOME (Chemin standard pour OpenJDK 17 sur Debian arm64)
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64
ENV PATH="${JAVA_HOME}/bin/:${PATH}"

USER airflow

# Copier les dépendances du projet
COPY requirements.txt /requirements.txt

# Installer les dépendances Python
RUN pip install --no-cache-dir -r /requirements.txt
