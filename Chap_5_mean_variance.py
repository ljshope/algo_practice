
import os
import pandas as pd
import numpy as np
from numpy.random import random, uniform, dirichlet, choice
from numpy.linalg import inv

from scipy.optimize import minimize

import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

np.random.seed(42)

cmap = sns.diverging_palette(10, 240, n = 9, as_cmap = True)

path = os.getcwd()

with pd.HDFStore(path+'/assets.h5') as store:
    sp500_stocks = store['sp500/stocks']    

with pd.HDFStore(path+'/assets.h5') as store:
    prices = (store['quandl/wiki/prices']
              .adj_close
              .unstack('ticker')
              .filter(sp500_stocks.index)
              .sample(n=30, axis=1))    
    
start = 2008
end = 2017

weekly_returns = prices.loc[f'{start}':f'{end}'].resample('W').last().pct_change().dropna(how='all')
weekly_returns = weekly_returns.dropna(axis=1)
weekly_returns.info()
stocks = weekly_returns.columns 

n_obs, n_assets = weekly_returns.shape
NUM_PF = 100000
x0 = uniform(0,1, n_assets)
x0 /= np.sum(np.abs(x0))

periods_per_year = round( weekly_returns.resample('A').size().mean() )

mean_returns = weekly_returns.mean()
cov_matrix = weekly_returns.cov()

precision_matrix = pd.DataFrame( inv(cov_matrix), index = stocks, columns = stocks )

treasury_10yr_monthly = (web.DataReader('DGS10', 'fred', start, end)
                         .resample('M')
                         .last()
                         .div(periods_per_year)
                         .div(100)
                         .squeeze())
rf_rate = treasury_10yr_monthly.mean()

def simulate_portfolios( mean_ret, cov, rf_rate = rf_rate, short = True ):
    alpha = np.full( shape = n_assets, fill_value = .05 )
    weights = dirichlet( alpha = alpha, size = NUM_PF )
    if short:
        weights *= choice( [-1, 1], size = weights.shape )
    returns = weights @mean_ret.values +1 
    returns = returns ** periods_per_year -1 
    std = (weights@weekly_returns.T).std(1)
    std *= np.sqrt( periods_per_year )
    sharpe = (returns-rf_rate)/std
    return pd.DataFrame( { 'Annualized std' : std, 
                           'Annualized Returns' : returns, 
                           'Sharpe Ratio': sharpe}), weights    

simul_perf, simul_wt = simulate_portfolios(mean_returns, cov_matrix, short=False)

ax = simul_perf.plot.scatter(x=0, y=1, c=2, cmap='Blues',
                             alpha=0.5, figsize=(14, 9), colorbar=True,
                             title=f'{NUM_PF:,d} Simulated Portfolios')

max_sharpe_idx = simul_perf.iloc[:,2].idxmax()
sd, r = simul_perf.iloc[max_sharpe_idx, :2 ].values
ax.scatter(sd, r, marker='*', color='darkblue', s=500, label='Max. Sharpe Ratio')

min_vol_idx = simul_perf.iloc[:, 0].idxmin()
sd, r = simul_perf.iloc[min_vol_idx, :2].values
ax.scatter(sd, r, marker='*', color='green', s=500, label='Min Volatility')
plt.legend(labelspacing=1, loc='upper left')
plt.tight_layout()

def portfolio_std( wt, rt = None, cov = None):
    return np.sqrt( wt@cov@wt *periods_per_year)

def portfolio_return( wt, rt = None, cov = None):
    return (wt@rt+1)**periods_per_year -1

def portfolio_performance( wt, rt = None, cov = None):
    sd = portfolio_std( wt, cov = cov)
    r = portfolio_return( wt, rt = rt)
    return r, sd 

def neg_sharpe_ratio( wt, rt, cov):
    r, sd = portfolio_performance( wt, rt, cov)
    return - ( r - rf_rate)/sd

weight_constraint = {'type':'eq', 'fun': lambda x: np.sum( np.abs(x))-1 }

def max_sharpe_ratio( mean_ret, cov, short = False):
    return minimize( fun = neg_sharpe_ratio,
                     x0 = x0,
                     args = (mean_ret, cov),
                     method = 'SLSQP',
                     bounds = ( ( -1 if short else 0, 1),)*n_assets,
                     constraints = weight_constraint,
                     options = { 'tol':1e-10, 'maxiter':1e4}
                     )

result = max_sharpe_ratio( mean_returns, cov_matrix, short = False)
max_sharpe_perf = portfolio_performance( result.x, mean_returns, cov_matrix)


   


