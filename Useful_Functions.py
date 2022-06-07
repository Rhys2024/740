import pandas as pd
import yfinance as yf
import fredapi as fa
from datetime import date, timedelta
#from Sector_Exposure import Rate_Exposures
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm


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
    
    gradient = pd.Series([(grad-target_rate[num])/15 for num, grad in enumerate(target_rate[15:])], index = [i for i in target.index[15:]]).dropna()
    
    gradient_means = gradient.rolling(lookback, center=False).mean()
    gradient_std = gradient.rolling(lookback, center=False).std()
    
    target_rate_means = target_rate.rolling(lookback, center=False).mean()
    target_rate_std = target_rate.rolling(lookback, center=False).std()
    standardized_g = pd.Series((gradient[lookback:] - gradient_means[lookback:])/(gradient_std[lookback:]))
    standardized_df = pd.Series((target_rate[lookback:] - target_rate_means[lookback:])/(target_rate_std[lookback:]))
    
    raw_scores = standardized_g + standardized_df
    
    scores = round(raw_scores, 0)
    
    return scores


# compare_group = None
# assert compare_group
def update_data():
    
    fred = fa.Fred('4fb0ce271d0f66f4b5b3904b4aaf1dd0')
    
    # "ten", "two",
    sector_names = ['Materials', 'Industrials', 'Consumer Discretionary', 
                        'Consumer Staples', 'Health Care',
                        'Financials', 'Information Technology', 
                        'Telecommunication Services', 'Utilities', 'Real Estate', 'Energy', 'Semiconductors', "Aerospace",
                        "SP-500", "Real Yield", "Yield Curve"]
    
    # "ten", "two",
    factor_names = ['Value', 'Quality', 'Size', 'Default', "Real Yield", "Yield Curve"]
    
    #if 'sector' in compare_group.lower():
        
    sec = yf.download([f"^SP500-{i}" for i in range(15,65,5)], start = '2010-01-04', progress=False)["Close"]
    sec['Energy'] = yf.download(["^GSPE"], start = '2010-01-04', progress=False)["Close"]
    sec['Semiconductors'] = yf.download(["SOXX"], start = '2010-01-04', progress=False)["Close"]
    sec['Aerospace'] = yf.download(["ITA"], start = '2010-01-04', progress=False)["Close"]
    sec['SP-500'] = yf.download("^GSPC", start = '2010-01-04', progress=False)["Close"]
    sec['Real Yield'] = fred.get_series('DFII10', observation_start = '2010-01-04', end = date.today())
    sec['Yield Curve'] = fred.get_series('T10Y2Y', observation_start = '2010-01-04', end = date.today())
        
    #if 'factor' in compare_group.lower():
        
    fact = pd.DataFrame()
    fact['Value'] = yf.download(["VLUE"], start = '2010-01-04', progress=False)["Close"]
    fact['Quality'] = yf.download(["QUAL"], start = '2010-01-04', progress=False)["Close"]
    fact['Size'] = yf.download(["SIZE"], start = '2010-01-04', progress=False)["Close"]
    fact['Default'] = yf.download(["FIBR"], start = '2010-01-04', progress=False)["Close"]
    fact['Real Yield'] = sec['Real Yield']
    fact['Yield Curve'] = sec['Yield Curve']
    
    sec.columns = sector_names
    fact.columns = factor_names
    
    #sectors = pd.DataFrame(sec.values, columns = sector_names, index = sec.index)
    
    #sec['ten'] = fred.get_series('DGS10', observation_start = '2010-01-04', end = date.today())
    #sec['two'] = fred.get_series('DGS2', observation_start = '2010-01-04', end = date.today())
    
    sec.dropna().to_csv("sectors_and_rates.csv")
    fact.dropna().to_csv("factors_and_rates.csv")
    
    #if 'sector' in compare_group.lower():
        #sec.columns = sector_names
        #return sec.dropna()
    
    #if 'factor' in compare_group.lower():
        #sec.columns = factor_names
        #return sec.dropna()


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