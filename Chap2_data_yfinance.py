
import pandas as pd
import yfinance as yf

symbol ='FB'
ticker = yf.Ticker(symbol)

pd.Series(ticker.info).head(20)

data = ticker.history(period='5d',
                      interval='1m',
                      start=None,
                      end=None,
                      actions=True,
                      auto_adjust=True,
                      back_adjust=False)
#data.info()

#ESG
ticker.sustainability

#analyst
#ticker.recommendations.info()
#ticker.recommendations.tail(10)

#ticker.options
expiration = ticker.options[0]
options = ticker.option_chain(expiration)
options.calls.iloc[0]
