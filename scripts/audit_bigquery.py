from google.cloud import bigquery
import pandas as pd
import os

def main():
    client = bigquery.Client()
    # On utilise le nom exact trouvé dans le script d'export
    dataset_id = 'Dataset_Strategy_Momentum'
    tables = ['gold_stock_features', 'gold_etf_features', 'gold_sp500_index_features']

    print(f"\n=== AUDIT BIGQUERY (Dataset: {dataset_id}) ===")
    print("-" * 85)
    print(f"{'Table':<25} | {'Lignes':<10} | {'Date Début':<15} | {'Date Fin':<15}")
    print("-" * 85)
    
    for table in tables:
        try:
            # Note: BigQuery est sensible à la casse
            query = f"SELECT count(*) as total, min(Date) as min_date, max(Date) as max_date FROM `{dataset_id}.{table}`"
            df = client.query(query).to_dataframe()
            if not df.empty:
                row = df.iloc[0]
                print(f"{table:<25} | {int(row['total']):<10} | {str(row['min_date']):<15} | {str(row['max_date']):<15}")
            else:
                print(f"{table:<25} | 0          | N/A             | N/A")
        except Exception as e:
            print(f"{table:<25} | ❌ Erreur: {str(e)[:45]}...")
    print("-" * 85)

if __name__ == "__main__":
    main()
