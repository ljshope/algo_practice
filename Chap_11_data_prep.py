import os
import numpy as np
import pandas as pd
import pandas_datareader.data as web

#from pyfinance.ols import PandasRollingOLS
#from talib import RSI, BBANDS, MACD, NATR, ATR

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

path = os.getcwd()

start = 2008
end = 2017

with pd.HDFStore(path+'/assets.h5') as store:
    prices = (store['quandl/wiki/prices']
          #    .loc[f'{start}':f'{end}']
              .filter(like='adj_')
              .dropna()
              .swaplevel()
              .rename(columns = lambda x:x.replace('adj_',''))
   #           .join(store['us_equities/stocks'].loc[:,['sector']])
              .dropna())

query = prices.index.get_level_values(1) >= pd.Timestamp('2008-01-01')
prices = prices[query]

#len( prices.index.unique('ticker') ) 
min_obs = 10*252
nobs = prices.groupby( level = 'ticker').size()
to_drop = nobs[ nobs < min_obs ].index
prices = prices.drop(to_drop, level='ticker')

prices.to_hdf('data.h5', 'us/equities/prices')

