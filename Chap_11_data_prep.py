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

prices['dollar_volume'] = prices.loc[:,'close'].mul(prices.loc[:,'volume'], axis = 0)
prices.dollar_volume /= 1e6
prices = prices.unstack('ticker')
data = (pd.concat([prices.dollar_volume.resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                   prices[last_cols].resample('M').last().stack('ticker')],
                  axis=1)
        .swaplevel()
        .dropna())

outlier_cutoff = 0.01
lags = [1, 3, 6, 12]
returns = []

for lag in lags:
    returns.append(data
                   .close
                   .unstack('ticker')
                   .sort_index()
                   .pct_change(lag)
                   .stack('ticker')
                   .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                          upper=x.quantile(1-outlier_cutoff)))
                   .add(1)
                   .pow(1/lag)
                   .sub(1)
                   .to_frame(f'return_{lag}m')
                   )
    
returns = pd.concat(returns, axis=1).swaplevel()
returns.info(null_counts=True)
cmap = sns.diverging_palette(10, 220, as_cmap=True)
sns.clustermap(returns.corr('spearman'), annot=True, center=0, cmap=cmap);

data = data.join( returns ).dropna()

min_obs =111
nobs = data.groupby(level='ticker').size()
to_drop = nobs[nobs < min_obs].index
data = data.drop(to_drop, level='ticker')

data['target'] = data.groupby(level='ticker')[f'return_1m'].shift(-1)
data = data.dropna()

with pd.HDFStore('data.h5') as store:
    store.put('us/equities/monthly', data)


