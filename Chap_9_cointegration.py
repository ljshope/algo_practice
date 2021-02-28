 
from time import time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.tree import  DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV

import numpy as np
from numpy.linalg import LinAlgError

from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR

import matplotlib.pyplot as plt
import seaborn as sns
import os

pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

path = os.getcwd() 
DATA_PATH = Path(path, 'stooq')
STORE = DATA_PATH / 'assets.h5'

critical_values = {0: {.9: 13.4294, .95: 15.4943, .99: 19.9349},
                   1: {.9: 2.7055, .95: 3.8415, .99: 6.6349}}

trace0_cv = critical_values[0][.95] # critical value for 0 cointegration relationships
trace1_cv = critical_values[1][.95] # critical value for 1 cointegration relationship

def remove_correlated_assets(df, cutoff=.99):
    corr = df.corr().stack()
    corr = corr[corr < 1]
    to_check = corr[corr.abs() > cutoff].index
    keep, drop = set(), set()
    for s1, s2 in to_check:
        if s1 not in keep:
            if s2 not in keep:
                keep.add(s1)
                drop.add(s2)
            else:
                drop.add(s1)
        else:
            keep.discard(s2)
            drop.add(s2)
    return df.drop(drop, axis=1)

def check_stationarity(df):
    results = []
    for ticker, prices in df.items():
        results.append([ticker, adfuller(prices, regression='ct')[1]])
    return pd.DataFrame(results, columns=['ticker', 'adf']).sort_values('adf')

def remove_stationary_assets(df, pval=.05):
    test_result = check_stationarity(df)
    stationary = test_result.loc[test_result.adf <= pval, 'ticker'].tolist()
    return df.drop(stationary, axis=1).sort_index()

def select_assets(asset_class='stocks', n=500, start=2010, end=2019):
    idx = pd.IndexSlice
    with pd.HDFStore(STORE) as store:
        df = (pd.concat([
            store[f'stooq/us/nasdaq/{asset_class}/prices'],
            store[f'stooq/us/nyse/{asset_class}/prices']
        ]).sort_index().loc[idx[:, str(start):str(end)], :])
        df = df.reset_index().drop_duplicates().set_index(['ticker', 'date'])
        df['dv'] = df.close.mul(df.volume)
        dv = df.groupby(level='ticker').dv.mean().nlargest(n=n).index
        df = (df.loc[idx[dv, :],
                     'close'].unstack('ticker').ffill(limit=5).dropna(axis=1))

    df = remove_correlated_assets(df)
    return remove_stationary_assets(df).sort_index()

#for asset_class, n in [('etfs', 500), ('stocks', 250)]:
#    df = select_assets(asset_class=asset_class, n=n)
#    df.to_hdf('data.h5', f'{asset_class}/close')

def get_ticker_dict():
    with pd.HDFStore(STORE) as store:
        return (pd.concat([
            store['stooq/us/nyse/stocks/tickers'],
            store['stooq/us/nyse/etfs/tickers'],
            store['stooq/us/nasdaq/etfs/tickers'],
            store['stooq/us/nasdaq/stocks/tickers']
        ]).drop_duplicates().set_index('ticker').squeeze().to_dict())
    
stocks = pd.read_hdf('data.h5', 'stocks/close')
etfs = pd.read_hdf('data.h5', 'etfs/close')
    
names = stocks.columns
pd.Series( names ).to_hdf('data.h5', 'tickers') 

corr = pd.DataFrame(index=stocks.columns)
for etf, data in etfs.items():
    corr[etf] = stocks.corrwith(data)

#cmap = sns.diverging_palette( 220, 10, as_cmap = True)
#sns.clustermap( corr, cmap= cmap, center = 0)

#stocks.shape

security = etfs['AAXJ.US'].loc['2010':'2020'] 
candidates = stocks.loc['2010':'2020']

security = security.div( security.iloc[0] )   
candidates = candidates.div( candidates.iloc[0] )
spreads = candidates.sub( security, axis = 0 ) 

n, m = spreads.shape
X = np.ones(shape = (n,2))
X[:, 1 ] = np.arange(1, n+1)

for candidate, prices in candidates.items():
    df = pd.DataFrame( { 's1': security, 's2': prices})
    var = VAR(df.values)
    lags = var.select_order()
    k_ar_diff = lags.selected_orders['aic']
    coint_johansen( df, det_order = 0, k_ar_diff = k_ar_diff)
    coint( security, prices, trend = 'c')[:2]
    coint( prices, security, trend = 'c')[:2]




