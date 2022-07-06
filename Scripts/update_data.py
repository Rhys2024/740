import numpy as np
import pandas as pd
from datetime import date, timedelta
import fredapi as fa
import yfinance as yf
import os, platform
import FileFinder.FileFinder as ff

#open('Private.json')

fred = fa.Fred('4fb0ce271d0f66f4b5b3904b4aaf1dd0')

#### Labels for update ####
#####################################################################################################################
sector_ids = {'Materials': '^SP500-15','Industrials': '^SP500-20','Consumer Discretionary': '^SP500-25', 
              'Consumer Staples': '^SP500-30','Health Care': '^SP500-35','Financials': '^SP500-40',
              'Technology': '^SP500-45','Telecomm': '^SP500-50','Utilities': '^SP500-55',
              'Real Estate': '^SP500-60','Energy': '^GSPE','sp_500': '^GSPC'}
#####################################################################################################################
factor_ids = {'Value' : 'VLUE', 'Growth' : 'IWF', 'Quality' : 'QUAL', 'Size' : 'SIZE', 'Default' : 'FIBR',
                'Volatility_Short' : 'UVXY', 'Momentum' : 'MTUM', 'beta_high_semiconductors' : 'XOP'}
#####################################################################################################################
secs_for_sector_etf_webpage = ["materials", "industrials", "consumer-discretionaries", "consumer-staples", 
                                "healthcare", "financials", "technology", "telecom", "utilities", "real-estate"]
important_features_for_sector_etf_webpage =["Symbol", "ETF Name", "Industry", "Previous Closing Price", "Beta", 
                                            "P/E Ratio", "YTD", "1 Month", "1 Year"]
#####################################################################################################################
macro_data = ['nominal_yield', 'inflation', 'cpi', 'real_yield', 'yield_curve', 'economic_uncertainty',
                'market_price', 'market_volume', 'market_volatility', 'volatility_expectation', 'russel_two_volatility', 
                'large_cap_volatility', 'oil', 'unemployment']

# , 'jobless_claims' 'weekly' : ['jobless_claims'], , 'jobless_claims' : 'ICSA'

macro_cats = {'fred' : {'daily' : ['nominal_yield', 'inflation', 'real_yield', 'yield_curve', 
                                   'economic_uncertainty', 'large_cap_volatility', 'volatility_expectation',
                                   'russel_two_volatility', 'oil'],
                        'monthly' : ['cpi', 'unemployment'],
                        'quarterly' : []},
                'yahoo' : {'market_data' :{'Volume' : ['market_volume'],
                                    'Close' : ['market_price'],
                                    'Volatility' : ['market_volatility']}
                            },
                }
fred_ids = {'nominal_yield' : 'DGS10', 'inflation' : 'T10YIE', 'cpi' : 'CPALTT01USM657N', 
            'real_yield' : 'DFII10', 'yield_curve' : 'T10Y2Y', 'economic_uncertainty' : 'USEPUINDXD', 'large_cap_volatility' : 'VXNCLS',
            'volatility_expectation' : 'VIXCLS', 'russel_two_volatility' : 'RVXCLS', 'oil' : 'DCOILWTICO', 'unemployment' : 'UNRATE'}
#####################################################################################################################

def update_dataframe(ids):
    
    starting_day = "2003-01-02"
    
    ####  DATA  ####
    df = pd.DataFrame()
    for ticker_name in ids:
        df[ticker_name] = yf.download(ids[ticker_name], start = starting_day, progress=False)["Close"]
    df = round(df.dropna(), 4)
    return df
    #######################################
    
def export_to_ReferenceFolder(df, name):
    
    path = ff.get_path(f'{name}.csv', 'Reference_Data')
    df.to_csv(path)
    

def update_data():
    
    starting_day_macro = "2003-01-02"
    
    sector_df = update_dataframe(sector_ids)
    factor_df = update_dataframe(factor_ids)
    
    
    #### MACRO DATA ####

    macro = pd.DataFrame()
    for cat in macro_data:
        print(cat)
        if cat in macro_cats['fred']['daily']:           
            macro[cat] = fred.get_series(fred_ids[cat], observation_start=starting_day_macro, end = date.today())

        elif cat in macro_cats['fred']['monthly']:
            monthly = fred.get_series(fred_ids[cat], observation_start=starting_day_macro, end = date.today())
            macro[cat] = monthly
            macro[cat] = macro[cat].ffill().replace(np.nan, monthly[0])
        
            '''
            elif cat in macro_cats['fred']['weekly']:
                weekly = fred.get_series(fred_ids[cat], observation_start=starting_day_macro, end = date.today())
                macro[cat] = weekly
                macro[cat] = macro[cat].replace(np.nan, weekly[0])
            '''
        
        # Clean up above to handle more shit
        
        
        elif cat in macro_cats['yahoo']['market_data']['Volume']:
                
            macro[cat] = yf.download('^IXIC', start = starting_day_macro, progress=False)['Volume']
                
        elif cat in macro_cats['yahoo']['market_data']['Close']:
            
            macro[cat] = yf.download('^IXIC', start = starting_day_macro, progress=False)['Close']
                
        elif cat in macro_cats['yahoo']['market_data']['Volatility']:
            
            #macro = macro.dropna()
            try:
                log_rets = pd.Series(np.log(macro['market_price']/macro['market_price'].shift()), index = macro.index)
            except:
                market_close = yf.download('^IXIC', start = starting_day_macro, progress=False)['Close']
                
                log_rets = pd.Series(np.log(market_close/market_close.shift()), index = macro.index)
                
            macro[cat] = log_rets.rolling(30).std()*252**.5
        
        else:
            pass 
    
    
    macro.index.name = 'Date'
    macro = round(macro.dropna(), 4)
    #######################################
    
    #### Export to Excel for use ####
    export_to_ReferenceFolder(macro, 'macro_data')
    export_to_ReferenceFolder(sector_df, 'sector_data')
    export_to_ReferenceFolder(factor_df, 'factor_data')
    
    sector_names = list(sector_ids.keys())
    sector_etfs = {}
    writer = pd.ExcelWriter(ff.get_path("Sector_ETF_Options.xlsx"), engine='xlsxwriter')
    for num, s in enumerate(secs_for_sector_etf_webpage):
        url = f"https://etfdb.com/etfs/sector/{s}/"
        sector_etfs[sector_names[num]] = pd.read_html(url)[0].iloc[:-1,:][important_features_for_sector_etf_webpage]
        sector_etfs[sector_names[num]].to_excel(writer, sheet_name = f"{sector_names[num]} ETFs")
    
    writer.save()
    #######################################
    

if __name__ == "__main__":
    #macro_path = get_path('macro_data.csv')
    #macro.to_csv('macro_data.csv')
    #print(macro_path)
    print('\nupdating data...\n\n')
    update_data()
    print('\nFinished updating data!\n\n')