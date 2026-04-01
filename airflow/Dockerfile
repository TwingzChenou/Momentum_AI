FROM apache/airflow:2.10.2-python3.10

USER root

# Installer Java 21 (Nécessaire pour Spark)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jdk-headless \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configurer JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin/:${PATH}"

USER airflow

# Copier les dépendances du projet
COPY requirements.txt /requirements.txt

# Installer les dépendances Python
RUN pip install --no-cache-dir -r /requirements.txt
