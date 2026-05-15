
import pandas as pd
import csv
import re
from datetime import datetime

def parse_github_history(input_path, output_path):
    print(f"📖 Parsing {input_path}...")
    
    ticker_data = {} # {ticker: {'start': date, 'end': date}}
    
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            current_date = row[0]
            tickers_raw = row[1].split(',')
            
            for t_raw in tickers_raw:
                # Gérer le format Ticker-YYYYMM
                match = re.match(r"([A-Z\.]+)-?(\d{6})?", t_raw)
                if match:
                    ticker = match.group(1)
                    exit_date_str = match.group(2)
                    
                    if ticker not in ticker_data:
                        ticker_data[ticker] = {'Date_start': current_date, 'Date_end': None}
                    
                    if exit_date_str:
                        # Convertir YYYYMM en YYYY-MM-01
                        exit_date = f"{exit_date_str[:4]}-{exit_date_str[4:6]}-01"
                        ticker_data[ticker]['Date_end'] = exit_date
    
    # Convertir en DataFrame
    records = []
    for ticker, dates in ticker_data.items():
        records.append({
            'Ticker': ticker,
            'Date_start': dates['Date_start'],
            'Date_end': dates['Date_end']
        })
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"✅ Extraction terminée : {len(df)} tickers trouvés. Sauvegardé dans {output_path}")

if __name__ == "__main__":
    parse_github_history(
        "/Users/forget/Desktop/Project_Momentum_AI/scratch/sp500_history_github.csv",
        "/Users/forget/Desktop/Project_Momentum_AI/scratch/sp500_history_cleaned.csv"
    )
