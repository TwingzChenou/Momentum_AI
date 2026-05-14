from pyspark.sql import functions as F
from pyspark.sql.window import Window

def calculate_technical_indicators(df, sma_fast_p=13, sma_slow_p=50, adx_p=9, atr_p=4, mom_p=20):
    """
    Calcule les indicateurs techniques optimisés pour la stratégie Momentum.
    Version Spark native pour gérer des millions de lignes d'historique.
    """
    # Fenêtre par Ticker ordonnée par Date
    window_spec = Window.partitionBy("Ticker").orderBy("Date")
    
    # 1. Moyennes Mobiles (SMA)
    df = df.withColumn("SMA_fast", F.avg("Close").over(window_spec.rowsBetween(-(sma_fast_p - 1), 0)))
    df = df.withColumn("SMA_slow", F.avg("Close").over(window_spec.rowsBetween(-(sma_slow_p - 1), 0)))
    
    # 2. Momentum
    df = df.withColumn("prev_close_mom", F.lag("Close", mom_p).over(window_spec))
    df = df.withColumn("Momentum_XM", (F.col("Close") - F.col("prev_close_mom")) / F.col("prev_close_mom"))
    
    # 3. ATR (Average True Range)
    df = df.withColumn("prev_close", F.lag("Close", 1).over(window_spec))
    df = df.withColumn("tr1", F.col("High") - F.col("Low"))
    df = df.withColumn("tr2", F.abs(F.col("High") - F.col("prev_close")))
    df = df.withColumn("tr3", F.abs(F.col("Low") - F.col("prev_close")))
    df = df.withColumn("TR", F.greatest("tr1", "tr2", "tr3"))
    
    df = df.withColumn("ATR", F.avg("TR").over(window_spec.rowsBetween(-(atr_p - 1), 0)))
    # ATR en pourcentage du prix (0-100)
    df = df.withColumn("ATR_pct", (F.col("ATR") / F.col("Close")) * 100)
    
    # 4. ADX (Average Directional Index) - Calcul complet Spark SQL
    df = df.withColumn("up_move", F.col("High") - F.lag("High", 1).over(window_spec))
    df = df.withColumn("down_move", F.lag("Low", 1).over(window_spec) - F.col("Low"))
    
    df = df.withColumn("dm_plus", F.when((F.col("up_move") > F.col("down_move")) & (F.col("up_move") > 0), F.col("up_move")).otherwise(0))
    df = df.withColumn("dm_minus", F.when((F.col("down_move") > F.col("up_move")) & (F.col("down_move") > 0), F.col("down_move")).otherwise(0))
    
    df = df.withColumn("dm_plus_smooth", F.avg("dm_plus").over(window_spec.rowsBetween(-(adx_p - 1), 0)))
    df = df.withColumn("dm_minus_smooth", F.avg("dm_minus").over(window_spec.rowsBetween(-(adx_p - 1), 0)))
    
    df = df.withColumn("di_plus", 100 * F.col("dm_plus_smooth") / F.col("ATR"))
    df = df.withColumn("di_minus", 100 * F.col("dm_minus_smooth") / F.col("ATR"))
    
    df = df.withColumn("di_sum", F.col("di_plus") + F.col("di_minus"))
    df = df.withColumn("dx", F.when(F.col("di_sum") != 0, 100 * F.abs(F.col("di_plus") - F.col("di_minus")) / F.col("di_sum")).otherwise(None))
    df = df.withColumn("ADX", F.avg("dx").over(window_spec.rowsBetween(-(adx_p - 1), 0)))
    
    # Nettoyage des colonnes temporaires
    cols_to_drop = ["prev_close_mom", "prev_close", "tr1", "tr2", "tr3", "TR", "up_move", "down_move", 
                    "dm_plus", "dm_minus", "dm_plus_smooth", "dm_minus_smooth", "di_plus", "di_minus", "di_sum", "dx", "ATR"]
    df = df.drop(*cols_to_drop)
    
    return df
