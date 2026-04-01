# Utiliser une image Python légère
FROM python:3.10-slim

# Éviter les questions interactives lors de l'install
ENV DEBIAN_FRONTEND=noninteractive

# Installer Java (JRE) pour faire tourner Spark
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-21-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configurer JAVA_HOME pour Docker (Sera surchargé si besoin via docker-compose)
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
ENV PATH=$JAVA_HOME/bin:$PATH

# Créer l'espace de travail
WORKDIR /app

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code du projet
COPY . .

# Définir les variables Python pour Spark
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# Exposer le port Streamlit
EXPOSE 8501

# Commande par défaut (Peut être surchargée par docker-compose)
CMD ["streamlit", "run", "src/streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
