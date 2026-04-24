import pandas as pd
alloc = [0, 0, 0.1, 0.1, 0.1, 0]
dates = pd.date_range('2023-01-01', periods=6)
allocations = pd.DataFrame({'AAPL': alloc}, index=dates)

last_alloc = allocations.iloc[-2] # day with 0.1
ticker = 'AAPL'

is_held = allocations[ticker] > 0
print("is_held:\n", is_held)

blocks = (is_held != is_held.shift()).cumsum()
print("blocks:\n", blocks)

last_block_id = blocks.iloc[-2] # simulating last day held
held_dates = allocations.index[blocks == last_block_id]
print("held_dates:\n", held_dates)
