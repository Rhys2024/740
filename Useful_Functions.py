import pandas as pd
import yfinance as yf
import fredapi as fa
from datetime import date, timedelta
#from Sector_Exposure import Rate_Exposures
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
sp500 = "SP-500"
ry = "Real Yield"
yc = "Yield Curve"

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
    
    if descrip:
        print("past month Real Yield, Yield Curve:\n")
        
    if window == "m":
        past_month_ry = round(scored_ry[-30:].mean(), 0)
        past_month_yc = round(scored_yc[-30:].mean(), 0)
        return (past_month_ry, past_month_yc)
    elif window == "d":
        return (scored_ry.iloc[-1], scored_yc.iloc[-1])

def get_scores(lookback, target):
    
    target_rate = target
    
    gradient = pd.Series([(grad-target_rate[num])/5 for num, grad in enumerate(target_rate[5:])], index = [i for i in target.index[5:]]).dropna()
    
    gradient_means = gradient.rolling(lookback, center=False).mean()
    gradient_std = gradient.rolling(lookback, center=False).std()
    
    target_rate_means = target_rate.rolling(lookback, center=False).mean()
    target_rate_std = target_rate.rolling(lookback, center=False).std()
    standardized_g = pd.Series((gradient[lookback:] - gradient_means[lookback:])/(gradient_std[lookback:]))
    standardized_df = pd.Series((target_rate[lookback:] - target_rate_means[lookback:])/(target_rate_std[lookback:]))
    
    raw_scores = standardized_g + standardized_df
    
    scores = round(raw_scores, 0)
    
    return scores.dropna()


def update_data():
    
    fred = fa.Fred('4fb0ce271d0f66f4b5b3904b4aaf1dd0')
    
    # 'Semiconductors', "Aerospace",
    sector_names = ['Materials', 'Industrials', 'Consumer Discretionary', 
                        'Consumer Staples', 'Health Care',
                        'Financials', 'Technology', 
                        'Telecomm', 'Utilities', 'Real Estate', 'Energy',
                        "SP-500", "Real Yield", "Yield Curve"]
    
    factor_names = ['Value', 'Quality', 'Size', 'Default', "Real Yield", "Yield Curve"]

    secs_for_sector_etf_webpage = ["materials", "industrials", "consumer-discretionaries", "consumer-staples", 
                                   "healthcare", "financials", "technology", "telecom", "utilities", "real-estate"]
    important_features_for_sector_etf_webpage =["Symbol", "ETF Name", "Industry", "Previous Closing Price", "Beta", "P/E Ratio", "YTD", "1 Month", "1 Year"]
    
    
    sectors = [f"^SP500-{i}" for i in range(15,65,5)]
    sectors.append("^GSPE")
    #sectors.append("SOXX")
    #sectors.append("ITA")
    sectors.append("^GSPC")
    sec = yf.download([f"^SP500-{i}" for i in range(15,65,5)], start = '2010-01-04', progress=False)["Close"]
    sec['Energy'] = yf.download(["^GSPE"], start = '2010-01-04', progress=False)["Close"]
    #sec['Semiconductors'] = yf.download(["SOXX"], start = '2010-01-04', progress=False)["Close"]
    #sec['Aerospace'] = yf.download(["ITA"], start = '2010-01-04', progress=False)["Close"]
    sec['SP-500'] = yf.download("^GSPC", start = '2010-01-04', progress=False)["Close"]
    sec['Real Yield'] = fred.get_series('DFII10', observation_start = '2010-01-04', end = date.today())
    sec['Yield Curve'] = fred.get_series('T10Y2Y', observation_start = '2010-01-04', end = date.today())
        
        
    fact = pd.DataFrame()
    fact['Value'] = yf.download(["VLUE"], start = '2010-01-04', progress=False)["Close"]
    fact['Quality'] = yf.download(["QUAL"], start = '2010-01-04', progress=False)["Close"]
    fact['Size'] = yf.download(["SIZE"], start = '2010-01-04', progress=False)["Close"]
    fact['Default'] = yf.download(["FIBR"], start = '2010-01-04', progress=False)["Close"]
    fact['Real Yield'] = sec['Real Yield']
    fact['Yield Curve'] = sec['Yield Curve']


    sector_etfs = {}
    writer = pd.ExcelWriter("Sector_ETF_Options.xlsx", engine='xlsxwriter')
    for num, s in enumerate(secs_for_sector_etf_webpage):
        url = f"https://etfdb.com/etfs/sector/{s}/"
        sector_etfs[sector_names[num]] = pd.read_html(url)[0].iloc[:-1,:][important_features_for_sector_etf_webpage]
        sector_etfs[sector_names[num]].to_excel(writer, sheet_name = f"{sector_names[num]} ETFs")
    
    writer.save()
        
    
    sec.columns = sector_names
    fact.columns = factor_names
    
    sec.dropna().to_csv("sectors_and_rates.csv")
    fact.dropna().to_csv("factors_and_rates.csv")
    


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


