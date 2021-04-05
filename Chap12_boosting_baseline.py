 
from pathlib import Path
import sys, os
from time import time
from tqdm import tqdm

from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd

import lightgbm as lgb
from catboost import Pool, CatBoostRegressor

from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

from alphalens.tears import (create_summary_tear_sheet,
                             create_full_tear_sheet)

from alphalens.utils import get_clean_factor_and_forward_returns

import matplotlib.pyplot as plt
import seaborn as sns

results_path = Path('results', 'baseline')
if not results_path.exists():
    results_path.mkdir(exist_ok=True, parents=True)

idx = pd.IndexSlice
np.random.seed(42)

DATA_STORE = 'assets.h5'
data = (pd.read_hdf('data.h5', 'model_data').sort_index()
                    .loc[idx[:,'2011':'2016'],:])

AAPL = data.loc[idx['AAPL',:]].loc[idx['2011':]]

base_params = dict(boosting='gbdt',
                   objective='regression',
                   verbose=-1)

# constraints on structure (depth) of each tree
max_depths = [6]#[2, 3, 5, 7]
num_leaves_opts = [2 ** i for i in max_depths]
min_data_in_leaf_opts = [5]#[250, 500, 1000]

# weight of each new tree in the ensemble
learning_rate_ops = [.01 ]#, .1, .3]

# random feature selection
feature_fraction_opts = [ .95]

param_names = ['learning_rate', 'num_leaves',
               'feature_fraction', 'min_data_in_leaf']

cv_params = list(product(learning_rate_ops,
                         num_leaves_opts,
                         feature_fraction_opts,
                         min_data_in_leaf_opts))
n_params = len(cv_params)
#print(f'# Parameters: {n_params}')

lookaheads = [1, 5, 21]
categoricals = ['year', 'month', 'weekday']
labels = sorted(AAPL.filter(like='_fwd').columns)
label_dict = dict(zip(lookaheads, labels))
for feature in categoricals:
    AAPL[feature] = pd.factorize(AAPL[feature], sort=True)[0]

features = AAPL.columns.difference(labels).tolist()

num_iterations = [10, 25, 50, 75] + list(range(100, 501, 100))
num_boost_round = num_iterations[-1]

train_lengths = [int(4.5 * 252) ]
test_lengths = [100]

test_params = list(product(lookaheads, train_lengths, test_lengths))

cv_preds = []
r = []
for lookahead, train_length, test_length in test_params:
    label = label_dict[lookahead]
    outcome_data = AAPL.loc[:, features + [label]].dropna()
    
    for p, param_vals in enumerate(cv_params):
        key = f'{lookahead}/{train_length}/{test_length}/' + '/'.join([str(p) for p in param_vals])
        params = dict(zip(param_names, param_vals))
        params.update(base_params)
    
    lgb_data = lgb.Dataset(data=outcome_data.drop(label, axis=1),
                           label=outcome_data[label],
                           categorical_feature=categoricals,
                           free_raw_data=False)
    train_idx = list( range( train_length- test_length) )
    lgb_train = lgb_data.subset(used_indices=train_idx,
                                params=params).construct()
      
    model = lgb.train(params=params,
                      train_set=lgb_train,
                      num_boost_round=num_boost_round,
                      verbose_eval=False)
    test_idx = list( range( len( outcome_data) - test_length, len(outcome_data)) )
    test_set = outcome_data.iloc[test_idx, :]
    X_test = test_set.loc[:, model.feature_name()]
    y_test = test_set.loc[:, label]
    y_pred = {str(n): model.predict(X_test, num_iteration=n) for n in num_iterations}
        
    a = np.array( y_test )
    b = np.array( y_pred["500"] )
    r.append( np.corrcoef(a,b)[0,1] )
    
#    cv_preds.append(y_test.to_frame('y_test').assign(**y_pred).assign( i = 0 ))
        
        # combine fold results
#    cv_preds = pd.concat(cv_preds).assign(**params)
 #       predictions.append(cv_preds)
        
        # compute IC per day
#    by_day = cv_preds.groupby(level='date')
#    ic_by_day = pd.concat([by_day.apply(lambda x: spearmanr(x.y_test, x[str(n)])[0]).to_frame(n)
#                           for n in num_iterations], axis=1)    
        
        