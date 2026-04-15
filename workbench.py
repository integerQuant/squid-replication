import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (10,6)

a = pd.read_parquet(r"data\clean\vix_futures\generic_contracts.parquet")

b = a[['trade_date', 'ux_symbol', 'settle']].set_index(['trade_date','ux_symbol']).unstack('ux_symbol')['settle']
b.columns.name = None
b.index = pd.to_datetime(b.index)

b.plot(figsize=(10, 6))

b.index.name = 'date'

bgs = pd.read_parquet("ux.parquet")

pd.concat([b,bgs['settle']], keys=['scrap', 'bg'])

b= b.loc['2006-03-14':]

bgs = bgs.reindex(b.index)

(bgs['settle'].pct_change() - b.pct_change()).dropna().loc['2013'].plot(figsize=(12,6))

bgs['settle'].plot()

b['UX1'].loc['2013'].plot()
bgs['settle']['UX1'].loc['2013'].plot()