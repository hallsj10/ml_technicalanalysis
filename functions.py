# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 08:57:14 2020

@author: halls35
"""
import numpy as np
import pandas as pd

from tia.bbg import LocalTerminal

def bloomberg_prices(tickers, start_dt, frequency, names):
    """
    Get Bloomberg price data
    """
    df=pd.DataFrame()
    rep1 = LocalTerminal.get_historical([tickers],['PX_LAST'], start=start_dt,             
                                        period=frequency)
    df=rep1.as_frame()
    df.columns = names
    return df

def fwd_returns(df, horizons=list):
    """
    Calculate forward returns for price series given list of time horizons
    """
    dts = df.index
    fwd_returns = dict()
    for h in horizons:
        fwd = df.pct_change(h).shift(-h)
        fwd_returns[h] = fwd
    out = pd.concat(fwd_returns.values(), axis=1, ignore_index=True)
    out.columns = horizons
    out.set_index(dts, inplace=True)
    
    return out
    
def trend_count(df):
    """
    Count consecutive positive or negative days
    """
    df1 = df.pct_change()
    df1.dropna(inplace=True)
    v = df1.values.flatten()
    dts = df1.index
    cc = []
    s = np.sign(v[0])
    cc.append(int(s))
    for d in range(1,len(df1)):
        
        if (v[d]> 0) & (v[d-1] > 0):
            s +=1
        elif (v[d] > 0) & (v[d-1] <= 0):
            s=1
        elif (v[d] < 0) & (v[d-1] <= 0):
            s-=1
        else:
            s=-1
        cc.append(int(s))
    out = pd.DataFrame(cc, index=dts, columns=['count']) 
    return out

def rsi(df, window):
    
    diff = df.diff(1).dropna()
    up_chg = 0 * diff
    down_chg = 0 * diff
    up_chg[diff > 0] = diff[diff > 0]
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    up_chg_avg   = up_chg.ewm(com=window-1 , min_periods=window).mean()
    down_chg_avg = down_chg.ewm(com=window-1 , min_periods=window).mean()  
    rs = abs(up_chg_avg/down_chg_avg)
    out = 100 - 100/(1+rs)
    out.columns = ['rsi']
    return out

def percent_moving_ave(df, window):
    df_ma = df.rolling(window).mean()
    out = df / df_ma - 1
    out.columns = ['percent_ma']
    return out

def realized_vol(df, window=int, freq=int):
    df1 = df.pct_change()
    out = df1.rolling(window).std()*np.sqrt(freq)
    out.columns = ['vol']
    return out