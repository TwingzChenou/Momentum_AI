import os
import sys
import pandas as pd
from datetime import datetime
sys.path.append(os.path.abspath('.'))
from src.common.setup_spark import create_spark_session
from src.strategy.backtest_engine import RegimeSwitchingMomentumBacktester
from config.config_spark import Paths

spark = create_spark_session(app_name="Test")
engine = RegimeSwitchingMomentumBacktester()

df_sp500 = engine.get_sp500_regime()
df_etf, df_stocks = engine.load_and_prep_data(spark, Paths.DATA_RAW_ETF_WEEKLY_GOLD, Paths.DATA_RAW_2B_WEEKLY_GOLD)

allocations = engine.simulate_portfolio(df_sp500, df_etf, df_stocks)
print("Allocations shape:", allocations.shape)

if not allocations.empty:
    perf_df = engine.generate_performance(allocations, df_etf, df_stocks, df_sp500)
    print("Perf_df shape:", perf_df.shape)
else:
    print("Allocations empty!")

spark.stop()
