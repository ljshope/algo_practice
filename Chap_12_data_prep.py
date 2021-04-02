
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
    
prices.volume /= 1e3 # make vol figures a bit smaller
prices.index.names = ['symbol', 'date']
metadata.index.name = 'symbol'

min_obs = 7*Year
nobs = prices.groupby(level = 'symbol').size()
keep= nobs[ nobs > min_obs ].index
prices = prices.loc[idx[keep,:],:]


metadata = metadata[~metadata.index.duplicated() & metadata.sector.notnull()]
metadata.sector = metadata.sector.str.lower().str.replace(' ', '_')

shared = ( prices.index.get_level_values('symbol').unique().intersection(metadata.index))  
metadata = metadata.loc[shared,:]
prices = prices.loc[idx[shared,:],:]
  
universe = metadata.marketcap.nlargest(500).index
prices = prices.loc[idx[universe,:],:]
metadata = metadata.loc[universe, :]

prices['dollar_vol'] = prices[['close', 'volume']].prod(1).div(1e3)    
dollar_vol_ma = ( prices.dollar_vol.unstack('symbol')
                 .rolling(window=21, min_periods =1).mean())

prices['dollar_vol_rank'] = (dollar_vol_ma
                            .rank(axis=1, ascending=False)
                            .stack('symbol')
                            .swaplevel())

prices['rsi'] = prices.groupby(level='symbol').close.apply(RSI)

def compute_bb(close):
    high, mid, low = BBANDS(close, timeperiod=20)
    return pd.DataFrame({'bb_high': high, 'bb_low': low}, index=close.index)

prices = (prices.join(prices
                      .groupby(level='symbol')
                      .close
                      .apply(compute_bb)))


prices['bb_high'] = prices.bb_high.sub(prices.close).div(prices.bb_high).apply(np.log1p)
prices['bb_low'] = prices.close.sub(prices.bb_low).div(prices.close).apply(np.log1p)

prices['NATR'] = prices.groupby(level='symbol', 
                                group_keys=False).apply(lambda x: 
                                                        talib.NATR(x.high, x.low, x.close))
def compute_atr(stock_data):
    df = ATR(stock_data.high, stock_data.low, 
             stock_data.close, timeperiod=14)
    return df.sub(df.mean()).div(df.std())

prices['ATR'] = (prices.groupby('symbol', group_keys=False)
                 .apply(compute_atr))

prices['PPO'] = prices.groupby(level='symbol').close.apply(talib.PPO)
def compute_macd(close):
    macd = MACD(close)[0]
    return (macd - np.mean(macd))/np.std(macd)
prices['MACD'] = (prices
                  .groupby('symbol', group_keys=False)
                  .close
                  .apply(compute_macd))

metadata.sector = pd.factorize(metadata.sector)[0].astype(int)
prices = prices.join(metadata[['sector']])
by_sym = prices.groupby(level='symbol').close
for t in T:
    prices[f'r{t:02}'] = by_sym.pct_change(t)


for t in T:
    prices[f'r{t:02}dec'] = (prices[f'r{t:02}']
                             .groupby(level='date')
                             .apply(lambda x: pd.qcut(x, 
                                                      q=10, 
                                                      labels=False, 
                                                      duplicates='drop')))
for t in T:
    prices[f'r{t:02}q_sector'] = (prices
                                  .groupby(['date', 'sector'])[f'r{t:02}']
                                  .transform(lambda x: pd.qcut(x, 
                                                               q=5, 
                                                               labels=False, 
                                                               duplicates='drop')))  
for t in [1, 5, 21]:
    prices[f'r{t:02}_fwd'] = prices.groupby(level='symbol')[f'r{t:02}'].shift(-t)

prices[[f'r{t:02}' for t in T]].describe()    
outliers = prices[prices.r01 > 1].index.get_level_values('symbol').unique()
prices = prices.drop(outliers, level='symbol')
prices['year'] = prices.index.get_level_values('date').year
prices['month'] = prices.index.get_level_values('date').month
prices['weekday'] = prices.index.get_level_values('date').weekday
prices.drop(['open', 'close', 'low', 'high', 'volume'], axis=1).to_hdf('data.h5', 'model_data')

