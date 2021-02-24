from datetime import datetime
import pandas as pd
import pandas_datareader.data as web
start = '2020'
end = '2022'

MSFT= web.DataReader('MSFT', 'yahoo', start=start, end=end)

SPX= web.DataReader('^GSPC', 'yahoo', start=start, end=end)
SPX_stooq = web.DataReader('^SPX', 'stooq', start=start, end=end)

SPX.Close.plot(figsize=(14,4))

#SPX.loc['2020-01-07']
#SPX.iloc[2]
#book = web.get_iex_book('AAPL')
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
symbols = get_nasdaq_symbols()