
import pandas as pd
import os


url = 'https://raw.githubusercontent.com/stefan-jansen/machine-learning-for-trading/master/data/us_equities_meta_data.csv'
df = pd.read_csv(url, index_col = 1)

path = os.getcwd()
DATA_STORE = path+'assets.h5'


with pd.HDFStore(DATA_STORE) as store:
    store.put('us_equities/stocks', df.set_index('ticker'))