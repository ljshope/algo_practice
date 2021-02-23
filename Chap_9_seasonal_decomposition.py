 
import pandas_datareader.data as web
import statsmodels.tsa.api as tsa
ind = web.DataReader('IPGMFN', 'fred', '1988', '2017-12').squeeze()
components = tsa.seasonal_decompose(ind, model = 'additive')

ts = ( ind.to_frame('Original')
       .assign(Trend=components.trend)
       .assign(Seasonality=components.seasonal)
       .assign(Residual=components.resid))
ts.plot(subplots=True, figsize=(14,8))