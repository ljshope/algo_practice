
from pathlib import Path
import warnings
from random import randint
import joblib
from itertools import product

import numpy as np
import pandas as pd

import shap
import lightgbm as lgb
from sklearn.inspection import (plot_partial_dependence, 
                                partial_dependence)

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

idx = pd.IndexSlice

wholedata = pd.read_hdf('data.h5', 'model_data').sort_index()
data = wholedata.loc[idx[:, '2012':'2017.6'], :]
test_set = wholedata.loc[idx[:, '2017.8':'2017.12.15'], :]

best_params = dict( train_length = 1134,
               test_length = 63, 
               learning_rate = 0.01,
               num_leaves = 128,
               feature_fraction = 0.95,
               min_data_in_leaf = 250,
               boost_rounds = 400 )

dates = sorted( data.index.get_level_values('date').unique() )
train_dates = dates[ -int( best_params['train_length']):]
data = data.loc[idx[:,train_dates],:]
labels = sorted( data.filter(like='_fwd').columns )
features = data.columns.difference(labels).tolist()
lookahead = 1
label = f'r{lookahead:02}_fwd'

categoricals = [ 'year', 'month', 'sector', 'weekday']

lgb_train = lgb.Dataset(data=data[features],
                       label=data[label],
                       categorical_feature=categoricals,
                       free_raw_data=False)

params = dict(boosting='gbdt', objective='regression', verbose=-1)
train_params = ['learning_rate', 'num_leaves', 'feature_fraction', 'min_data_in_leaf']
for val in train_params:
    params[val] = best_params[val] 
    
for p in ['min_data_in_leaf', 'num_leaves']:
    params[p] = int(params[p])

lgb_model = lgb.train(params=params,
                  train_set=lgb_train,
                  num_boost_round=int(best_params['boost_rounds']))

num_iterations = list(range(100, 401, 100))
test_set = test_set.dropna()
X_test = test_set.loc[:, lgb_model.feature_name()]
y_test = test_set.loc[:, label]
y_pred = {str(n): lgb_model.predict(X_test, num_iteration=n) for n in num_iterations}
        
a = np.array( y_test )
b = np.array( y_pred["400"] )
r =  np.corrcoef(a,b)[0,1]

def get_feature_importance(model, importance_type='split'):
    fi = pd.Series(model.feature_importance(importance_type=importance_type), 
                   index=model.feature_name())
    return fi/fi.sum()

feature_importance = (get_feature_importance(lgb_model).to_frame('Split').
                      join(get_feature_importance(lgb_model, 'gain').to_frame('Gain')))
(feature_importance
 .nlargest(20, columns='Gain')
 .sort_values('Gain', ascending=False)
 .plot
 .bar(subplots=True,
      layout=(2, 1),
      figsize=(14, 6),
      legend=False,
      sharey=True,
      rot=0))
plt.suptitle('Normalized Importance (Top 20 Features)', fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=.9)
#plt.savefig('figures/lgb_fi', dpi=300)



X = data[features].sample(n=1000)

# load JS visualization code to notebook
shap.initjs()

# explain the model's predictions using SHAP values
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X=X)

shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
#plt.savefig('figures/shap_dots', dpi=300)

shap.summary_plot(shap_values, X, plot_type="bar",show=False)
plt.tight_layout()
#plt.savefig('figures/shap_bar', dpi=300)

