

from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style('whitegrid')
idx = pd.IndexSlice

#path = os.getcwd() + '\\data\\nasdaq100'
path = os.getcwd() + '\\data\\test_example'
nasdaq_path = Path( path )

tcols = ['openbartime', 'firsttradetime',
         'highbidtime', 'highasktime', 'hightradetime',
         'lowbidtime', 'lowasktime', 'lowtradetime',
         'closebartime', 'lasttradetime']

drop_cols = ['unknowntickvolume',
             'cancelsize',
             'tradeatcrossorlocked']

keep = ['firsttradeprice', 'hightradeprice', 'lowtradeprice', 'lasttradeprice', 
        'minspread', 'maxspread',
        'volumeweightprice', 'nbboquotecount', 
        'tradeatbid', 'tradeatbidmid', 'tradeatmid', 'tradeatmidask', 'tradeatask', 
        'volume', 'totaltrades', 'finravolume', 
        'finravolumeweightprice', 
        'uptickvolume', 'downtickvolume', 
        'repeatuptickvolume', 'repeatdowntickvolume', 
        'tradetomidvolweight', 'tradetomidvolweightrelative']

columns = {'volumeweightprice'          : 'price',
           'finravolume'                : 'fvolume',
           'finravolumeweightprice'     : 'fprice',
           'uptickvolume'               : 'up',
           'downtickvolume'             : 'down',
           'repeatuptickvolume'         : 'rup',
           'repeatdowntickvolume'       : 'rdown',
           'firsttradeprice'            : 'first',
           'hightradeprice'             : 'high',
           'lowtradeprice'              : 'low',
           'lasttradeprice'             : 'last',
           'nbboquotecount'             : 'nbbo',
           'totaltrades'                : 'ntrades',
           'openbidprice'               : 'obprice',
           'openbidsize'                : 'obsize',
           'openaskprice'               : 'oaprice',
           'openasksize'                : 'oasize',
           'highbidprice'               : 'hbprice',
           'highbidsize'                : 'hbsize',
           'highaskprice'               : 'haprice',
           'highasksize'                : 'hasize',
           'lowbidprice'                : 'lbprice',
           'lowbidsize'                 : 'lbsize',
           'lowaskprice'                : 'laprice',
           'lowasksize'                 : 'lasize',
           'closebidprice'              : 'cbprice',
           'closebidsize'               : 'cbsize',
           'closeaskprice'              : 'caprice',
           'closeasksize'               : 'casize',
           'firsttradesize'             : 'firstsize',
           'hightradesize'              : 'highsize',
           'lowtradesize'               : 'lowsize',
           'lasttradesize'              : 'lastsize',
           'tradetomidvolweight'        : 'volweight',
           'tradetomidvolweightrelative': 'volweightrel'}

def extract_and_combine_data():
    path = nasdaq_path / '1min_taq'

    data = []
    # ~80K files to process
    for i, f in tqdm(enumerate(list(nasdaq_path.glob('*/*.csv.gz')))):
        data.append(pd.read_csv(f, parse_dates=[['Date', 'TimeBarStart']])
                    .rename(columns=str.lower)
                    .drop(tcols + drop_cols, axis=1)
                    .rename(columns=columns)
                    .set_index('date_timebarstart')
                    .sort_index()
                    .between_time('9:30', '16:00')
                    .set_index('ticker', append=True)
                    .swaplevel()
                    .rename(columns=lambda x: x.replace('tradeat', 'at')))
    data = pd.concat(data).apply(pd.to_numeric, downcast='integer')
  #  print(data.info(show_counts=True))
    data.to_hdf(nasdaq_path / 'algoseek.h5', 'min_taq')

extract_and_combine_data()
df = pd.read_hdf(nasdaq_path / 'algoseek.h5', 'min_taq')
number = len(df.index.unique('ticker'))

constituents = (df.groupby([df.index.get_level_values('date_timebarstart').date, 'ticker'])
                .size()
                .unstack('ticker')
                .notnull()
                .astype(int)
                .replace(0, np.nan))

