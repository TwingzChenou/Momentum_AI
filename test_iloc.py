import pandas as pd
df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
col_idx = df.columns.get_loc('a')
df.iloc[0, col_idx] = 100.0
print(df)

df_empty = pd.DataFrame(columns=['a', 'b'])
try:
    df_empty.iloc[0, df_empty.columns.get_loc('a')] = 100.0
except Exception as e:
    print("Empty error:", repr(e))
