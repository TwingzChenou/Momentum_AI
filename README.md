# Momentum AI: Regime-Switching Strategy with Optuna & MLflow

Momentum AI is an advanced algorithmic trading platform designed to execute a **regime-switching momentum strategy**. It dynamically allocates capital between S&P 500 stocks and defensive ETFs based on market conditions, optimized using Bayesian search and tracked via industrial-grade MLOps tools.

## 🎯 Project Goal

The primary objective of this project is to outperform the S&P 500 benchmark by:
1.  **Exploiting Momentum**: Selecting the top-performing assets in a bullish market.
2.  **Regime Switching**: Automatically shifting to defensive ETFs or Cash when the broader market (S&P 500) shows signs of weakness (Bear regime).
3.  **Automated Optimization**: Using machine learning (Optuna) to find the best possible hyperparameters (SMA windows, ADX thresholds, momentum periods) to maximize the Calmar.

---

## 🏗️ Architecture & Tech Stack

The project follows a modern data engineering and MLOps architecture:

-   **Data Storage**: Google BigQuery (organized in Bronze, Silver, and Gold layers).
-   **Processing**: Apache Spark (PySpark) for distributed data transformation and feature engineering.
-   **Optimization Engine**: Optuna for Bayesian hyperparameter optimization.
-   **Experiment Tracking**: MLflow for logging trials, parameters, and versioning the "Champion" model.
-   **Orchestration**: Apache Airflow (Dockerized) for automated weekly data updates and re-optimization.
-   **Dashboard**: Streamlit for real-time backtest visualization and performance monitoring.

---

## ⚙️ How It Works

### 1. Data Pipeline (Medallion Architecture)
*   **Bronze**: Raw data from Yahoo Finance (Stocks, ETFs, S&P 500).
*   **Silver**: Cleaned and normalized daily data.
*   **Gold**: Weekly aggregated features including SMA (Simple Moving Averages), ADX (Trend Strength), and ATR (Volatility).

### 2. Strategy Logic
The core engine (`backtest_engine.py`) follows these rules:
*   **Market Regime**: If S&P 500 Price > SMA_slow and SMA_fast > SMA_slow, the market is in a **Bull** regime.
*   **Bull Strategy**: Invests in the **Top 10** stocks with the highest momentum, filtered by ADX (strength) and ATR (volatility).
*   **Bear Strategy**: Invests in **Top 2** defensive ETFs (e.g., Gold, Treasury Bonds) or stays in **Cash**.
*   **Stop-Loss**: Daily/Weekly monitoring of the price relative to the SMA_slow to exit individual positions early.

### 3. Optimization Loop
The `strategy_optimizer.py` script runs an Optuna study:
*   It explores 11+ hyperparameters (e.g., SMA windows from 5 to 75 days).
*   Each trial runs a full historical backtest.
*   The best performing set of parameters is promoted as the **Champion** in MLflow.

---

## 🚀 Getting Started

### Prerequisites
*   Docker & Docker-Compose
*   Google Cloud Platform (GCP) Credentials for BigQuery

### Installation
1.  Clone the repository.
2.  Configure your environment variables in `.env`.
3.  Start the infrastructure:
    ```bash
    docker-compose up -d
    ```

### Running a Backtest
Access the Streamlit Dashboard (usually at `http://localhost:8501`) to:
*   Load the latest **Champion** config from MLflow.
*   Run a simulation on historical data.
*   Visualize Equity curves, Drawdowns, and Portfolio composition.

---

## 📊 Key Metrics Tracked
*   **CAGR**: Compound Annual Growth Rate.
*   **Max Drawdown**: The largest peak-to-trough decline.
*   **Sharpe Ratio**: Risk-adjusted return.
*   **Calmar Ratio**: Annual return relative to max drawdown (the primary optimization metric).

---

## 🛠️ Maintenance & Automation
The project includes Airflow DAGs to automate the entire lifecycle:
*   `dag_silver`: Weekly data ingestion and cleaning.
*   `strategy_optimization_weekly`: Automated re-tuning of the strategy every weekend to adapt to new market conditions.
