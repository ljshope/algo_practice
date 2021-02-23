 
import pandas_datareader.data as web
import statsmodels.tsa.api as tsa
ind = web.DataReader('IPGMFN', 'fred', '1988', '2017-12').squeeze()
components = tsa.seasonal_decompose(ind, model = 'additive')

ts = ( ind.to_frame('Original')
       .assign(Trend=components.trend)
       .assign(Seasonality=components.seasonal)
       .assign(Residual=components.resid))
ts.plot(subplots=True, figsize=(14,8))

df = web.DataReader(name='SP500', data_source='fred', start=2009).squeeze().to_frame('close')
spx = web.DataReader('SP500', 'fred', 2009, 2020).squeeze().to_frame('close')
