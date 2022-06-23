from matplotlib.style import available
import pandas as pd
import yfinance as yf
import fredapi as fa
import datetime
from datetime import date, timedelta
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import scipy.stats as si
from itertools import chain

######################################
mat = "Materials"
ind = "Industrials"
cd = "Consumer Discretionary"
cs = "Consumer Staples"
hc = "Health Care"
fin = "Financials"
tech = "Technology"
comm = "Telecomm"
ut = "Utilities"
re = "Real Estate"
en = "Energy"
semiconductors = "Semiconductors"
aero = "Aerospace"
sp500 = "sp_500"
######################################
nom = "nominal_yield"
inf = 'inflation'
ry = "real_yield"
yc = "yield_curve"
nom = 'nominal_yield'
uncertain = 'economic_uncertainty'
market_price = 'market_price'
mkt_volume = 'market_volume'
mkt_volatility = 'market_volatility'
cpi = 'cpi'
######################################

def beta(data_2_columns):
    
    log_returns = np.log(data_2_columns/data_2_columns.shift())
    
    return (log_returns.cov() / log_returns.var())[log_returns.columns[0]].iloc[1]


def beta_asset_to_index(asset_to_index):
    
    assert isinstance(asset_to_index, np.ndarray), "Make sure input data is a Numpy Array !!"
    assert asset_to_index.shape[1] == 2, "Make sure input data has 2 columns !!"
    
    asset_to_index = np.vstack((asset_to_index[:,0], asset_to_index[:,1]))
    
    #log_returns = np.log(asset_to_index/asset_to_index.shift()).values
    
    return np.cov(asset_to_index)[0][1] / np.var(asset_to_index)


def current_rates(window = "m", descrip = True):
    
    assert window == "m" or window == "d" or "mon" in window.lower() or window.lower() == "day", "window should be 'd' or 'm'"
    
    fred = fa.Fred('4fb0ce271d0f66f4b5b3904b4aaf1dd0')
    
    real_rate = fred.get_series('DFII10', observation_start = date.today() + timedelta(days= -600) , end = date.today())
    yield_curve = fred.get_series('T10Y2Y', observation_start = date.today() + timedelta(days= -600) , end = date.today())
    
    scored_ry = get_scores(252, real_rate.dropna())
    scored_yc = get_scores(252, yield_curve.dropna())
        
    if window == "m":
        if descrip:
            print("past month Real Yield, Yield Curve:\n")
        past_month_ry = round(scored_ry[-30:].mean(), 0)
        past_month_yc = round(scored_yc[-30:].mean(), 0)
        return (past_month_ry, past_month_yc)
    elif window == "d":
        if descrip:
            print("current day Real Yield, Yield Curve:\n")
        return (scored_ry.iloc[-1], scored_yc.iloc[-1])

def get_scores(lookback, target):
    
    gradient = target.diff(10)

    # min_periods = 252, window = len(ry)-1).mean()
    regular_scores = (target - target.rolling(lookback).mean()) / target.rolling(lookback).std()
    gradient_scores = (gradient - gradient.rolling(lookback).mean()) / gradient.rolling(lookback).std()
    
    raw_scores = np.round((regular_scores + gradient_scores).dropna(),3)
    
    scores = np.round((regular_scores + gradient_scores).dropna(),0).astype('int32')
    score_counts = scores.value_counts()

    significant = score_counts.loc[ score_counts >= 10 ]
    minimum = min(significant.index)
    maximum = max(significant.index)
    
    scores = scores.clip(lower=minimum, upper=maximum)

    return [scores, raw_scores]


def get_monthly_data(data):
        
    return data.groupby(pd.PeriodIndex(data.index, freq="M"))[[i for i in data.columns]].mean()


def is_stationary(series):
    
    return adfuller(series)[1] < .05


def VAR(df_of_returns, weights, n_days, current_value_of_portfolio, confidence_level = .05):
    
    returns = df_of_returns.pct_change()
    cov_returns = returns.cov()
    avg_rets = returns.mean()
    
    port_mean = avg_rets.dot(weights)
    port_stdev = np.sqrt(weights.T.dot(cov_returns).dot(weights))
    
    # Calculate mean of investment
    mean_investment = (1+port_mean) * current_value_of_portfolio
                
    # Calculate standard deviation of investmnet
    stdev_investment = current_value_of_portfolio * port_stdev

    cutoff1 = norm.ppf(confidence_level, mean_investment, stdev_investment)

    var_1d1 = current_value_of_portfolio - cutoff1

    return np.round(var_1d1*np.sqrt(n_days), 2)


def barplot_1d(pandas_series, stds_series = None, size = (12,8)):
    
    plt.figure(figsize=(size))
    if isinstance(stds_series, pd.core.series.Series):
        plt.bar(pandas_series.index, pandas_series, yerr = stds_series, capsize = 8)
    else:
        plt.bar(pandas_series.index, pandas_series)
    plt.show()
    

def vega_div(S, K, T, r, q, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    #q: continuous dividend rate
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    vega = 1 / np.sqrt(2 * np.pi) * S * np.exp(-q * T) * np.exp(-(d1 ** 2) * 0.5) * np.sqrt(T)
    
    return vega

def vega(S, K, T, r, sigma):

    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    vega = S * si.norm.cdf(d1, 0.0, 1.0) * np.sqrt(T)
    
    return vega


