# Machine Learning and Technical Analysis for Signal Generation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions import bloomberg_prices, trend_count, fwd_returns, rsi, \
percent_moving_ave, realized_vol

from regression_models import regression_predictions, regression_results

# Bloomberg query parameters
#tickers ='SPUSTTTR Index' # 10-year Futures TR Index
tickers ='SPY Equity'
start_dt = '1/1/2000'
frequency = 'DAILY' 
names = ['SPX']

#Load Bloomberg Data (or use another dataframe of price data)
df = bloomberg_prices(tickers, start_dt, frequency, names)

# Response Variables 
df_fwd = fwd_returns(df, horizons=[1,3,5])
y = df_fwd.iloc[:,0]

# Engineered Features
count_pos_neg_days = trend_count(df)
rsi =  rsi(df, 14)
perc_ma = percent_moving_ave(df, 50)
vol = realized_vol(df, window=60, freq=252)
df_xs = pd.merge(count_pos_neg_days, rsi, left_index=True, right_index=True)
df_xs = pd.merge(df_xs, perc_ma, left_index=True, right_index=True)
df_xs = pd.merge(df_xs, vol, left_index=True, right_index=True)
df_xs.dropna(inplace=True)
print(df_xs.corr())
x = df_xs # to have a copy

df_inputs = pd.merge(y, x, left_index=True, right_index=True)
split = 500
y_train = df_inputs.iloc[:-split,0]
x_train = df_inputs.iloc[:-split,1:]
y_test = df_inputs.iloc[-split:-1,0]
x_test = df_inputs.iloc[-split:-1,1:]
df_predict = regression_predictions(x_train, y_train, x_test, y_test)
residuals, rmse = regression_results(y_test, df_predict)


