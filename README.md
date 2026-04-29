# 🚀 Momentum AI: Algorithmic Strategy with Optuna & MLflow

Momentum AI est une plateforme de trading algorithmique avancée conçue pour exécuter une **stratégie de momentum à changement de régime**. Elle alloue dynamiquement du capital entre les actions du S&P 500 et des ETFs défensifs, optimisée par recherche Bayésienne et monitorée via MLflow.

---

## 🌟 Nouveautés & Améliorations Récentes

-   **🌐 Pont Réseau Docker-Hôte** : Intégration transparente entre le dashboard Streamlit (Docker) et le serveur MLflow (Mac) via `host.docker.internal`.
-   **📈 Profondeur Historique** : Extension de l'ingestion des données Bronze jusqu'au **1er janvier 2025**, offrant un recul critique pour les backtests.
-   **🛡️ Observabilité Totale** : Instrumentation complète de la pipeline (Bronze, Silver, Gold, Optimisation) avec `loguru` (tracking du temps d'exécution, volumétrie des données et alertes de qualité).
-   **🏆 Sélection Automatique du Champion** : Algorithme de recherche intelligent identifiant le dernier bilan d'optimisation (`Opt_...`) parmi des milliers d'essais techniques.

---

## 🏗️ Architecture & Stack Technique

-   **Data Storage** : Google BigQuery (Architecture Medallion : Bronze, Silver, Gold).
-   **Processing** : Apache Spark (PySpark) pour les transformations distribuées.
-   **Optimization Engine** : Optuna (TPE Sampler) pour la recherche d'hyperparamètres.
-   **Experiment Tracking** : MLflow pour le versioning du modèle "Champion" et le suivi des essais.
-   **Orchestration** : Apache Airflow pour l'automatisation hebdomadaire.
-   **Dashboard** : Streamlit pour la visualisation des performances et le backtest interactif.

---

## ⚙️ Fonctionnement de la Pipeline

### 1. Pipeline de Données (Medallion)
*   **Bronze** : Ingestion incrémentielle via Yahoo Finance pour les 503 membres actuels de l'index. **Début : 01/01/2025**.
*   **Silver** : Agrégation hebdomadaire (W-FRI) et filtrage strict basé sur les périodes d'inclusion historique dans le S&P 500.
*   **Gold** : Calcul des indicateurs techniques avancés (SMA, ADX, ATR) stockés dans BigQuery.

### 2. Logique de Stratégie
*   **Régime Bull** : Investissement dans le **Top N** des actions ayant le meilleur momentum, filtrées par volatilité (ATR) et force de tendance (ADX).
*   **Régime Bear** : Bascule automatique vers des ETFs de couverture (Or, Obligations) ou mise en Cash.

### 3. Observabilité & Logs
Chaque tâche est monitorée en temps réel. Exemple de log généré :
```bash
2026-04-29 13:45:00 | INFO    | 📋 503 tickers nécessitent une mise à jour.
2026-04-29 13:45:10 | SUCCESS | 💾 Sauvegarde terminée avec succès en 12.45s
2026-04-29 13:45:11 | INFO    | 🧪 Essai 15/50 | Score: 0.9142 | Meilleur: 0.9250
```

---

## 🚀 Démarrage Rapide

### Installation
1.  **Configuration** : Remplissez votre fichier `.env` avec vos accès GCP et MLflow.
2.  **Lancement de l'infrastructure** :
    ```bash
    docker-compose up -d
    ```
3.  **Lancement du Dashboard** : Accédez à `http://localhost:8501`.

### Utilisation du Dashboard
-   Le bouton **"Load Champion Config"** interroge MLflow pour récupérer les meilleurs paramètres (SMA, Momentum, etc.) trouvés par la dernière DAG d'optimisation.
-   Lancez la simulation pour visualiser la courbe d'équité, le Max Drawdown et le Ratio de Calmar.

---

## 📊 Métriques Clés d'Optimisation
*   **CAGR** : Taux de croissance annuel composé.
*   **Max Drawdown** : La perte maximale historique.
*   **Ratio de Calmar** : Notre métrique cible (CAGR / Max Drawdown).
*   **Sharpe Ratio** : Performance ajustée au risque.
