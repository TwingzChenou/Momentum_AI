FROM apache/airflow:2.10.2-python3.10

USER root

# 1. Créer un dossier universel pour les JARs et donner les droits à airflow
RUN mkdir -p /opt/airflow/jars && chown -R airflow: /opt/airflow/jars

# 2. Installer Java 17 et procps (Indispensable pour Spark/PySpark)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    procps \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. Configurer JAVA_HOME (Indispensable pour que Spark trouve Java)
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64
ENV PATH="${JAVA_HOME}/bin/:${PATH}"

USER airflow

# Copier les dépendances du projet
COPY requirements.txt /requirements.txt

# Installer les dépendances Python
RUN pip install --no-cache-dir -r /requirements.txt

# 4. Télécharger le connecteur GCS "Fat JAR" officiel dans notre nouveau dossier
USER root
RUN curl -sL -o /opt/airflow/jars/gcs-connector-hadoop3-latest.jar \
    https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar && \
    chown airflow: /opt/airflow/jars/gcs-connector-hadoop3-latest.jar

USER airflow
