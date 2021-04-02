
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import talib
from talib import RSI, BBANDS, MACD, ATR
import os 

Month = 21
Year = 12*Month

Start = '2010-01-01'
End = '2017-12-31'

idx = pd.IndexSlice

percentiles = [ .001, .01, .02, .03, .04, .05 ]
percentiles += [ 1-p for p in percentiles[::-1]]

T = [1,5,10,21,42,63]

#df = (pd.read_csv('wiki_prices.csv',
#                 parse_dates=['date'],
#                 index_col=['date', 'ticker'],
#                 infer_datetime_format=True)
#     .sort_index())

path = os.getcwd()
DATA_STORE = path+'assets.h5'

#with pd.HDFStore(DATA_STORE) as store:
#    store.put('quandl/wiki/prices', df)

#DATA_STORE = '../assets.h5'
ohlcv = ['adj_open', 'adj_close', 'adj_low', 'adj_high', 'adj_volume']
with pd.HDFStore(DATA_STORE) as store:
    prices = (store['quandl/wiki/prices']
              .loc[idx[Start:End, :], ohlcv] # select OHLCV columns from 2010 until 2017
              .rename(columns=lambda x: x.replace('adj_', '')) # simplify column names
              .swaplevel()
              .sort_index())
    metadata = (store['us_equities/stocks'].loc[:, ['marketcap', 'sector']])