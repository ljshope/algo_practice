 
from pathlib import Path
import requests
from io import BytesIO
from zipfile import ZipFile, BadZipFile
import os

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from sklearn.datasets import fetch_openml
from tqdm import tqdm

pd.set_option('display.expand_frame_repr', False)

DATA_STORE = Path('stooq/assets.h5')

stooq_path = Path('stooq') 
if not stooq_path.exists():
    stooq_path.mkdir()
    
data_path = os.getcwd() + '\\data\\stooq\\data'
data_path = Path(data_path)    


def get_stooq_prices_and_tickers(frequency='daily',
                                 market='us',
                                 asset_class='nasdaq etfs'):
    prices = []
    
    if frequency in ['5 min', 'hourly']:
        parse_dates = [['date', 'time']]
        date_label = 'date_time'
    else:
        parse_dates = ['date']
        date_label = 'date'
    names = ['ticker', 'freq', 'date', 'time', 
             'open', 'high', 'low', 'close','volume', 'openint']
    
    usecols = ['ticker', 'open', 'high', 'low', 'close', 'volume'] + parse_dates
    path = data_path / frequency / market / asset_class
    if 'stock' in asset_class:
        dir = '**/*.txt'
    else:
        dir = '*.txt'
    files = path.glob(dir)
    for i, file in enumerate(files, 1):
        if i % 500 == 0:
            print(i)
        try:
                df = (pd.read_csv(
                    file,
                    names=names,
                    usecols=usecols,
                    header=0,
                    parse_dates=parse_dates))
                prices.append(df)
        except pd.errors.EmptyDataError:
                print('\tdata missing', file.stem)
                file.unlink()

    prices = (pd.concat(prices, ignore_index=True)
              .rename(columns=str.lower)
              .set_index(['ticker', date_label])
              .apply(lambda x: pd.to_numeric(x, errors='coerce')))
    return prices
    
markets = {#'jp': ['tse stocks'],
           'us': ['nasdaq etfs', 'nasdaq stocks', 'nyse etfs', 'nyse stocks', 'nysemkt stocks']
          }
frequency = 'daily'

idx = pd.IndexSlice
for market, asset_classes in markets.items():
    for asset_class in asset_classes:
        print(f'\n{asset_class}')
        prices = get_stooq_prices_and_tickers(frequency=frequency, 
                                                       market=market, 
                                                       asset_class=asset_class)
        
        prices = prices.sort_index().loc[idx[:, '2000': '2021'], :]
        names = prices.index.names
        prices = (prices
                  .reset_index()
                  .drop_duplicates()
                  .set_index(names)
                  .sort_index())
        
        print('\nNo. of observations per asset')
        print(prices.groupby('ticker').size().describe())
        key = f'stooq/{market}/{asset_class.replace(" ", "/")}/'
        
        print(prices.info(null_counts=True))
        
        prices.to_hdf(DATA_STORE, key + 'prices', format='t')
    
    #    tickers.to_hdf(DATA_STORE, key + 'tickers', format='t')        
            
            