import numpy as np
import pandas as pd
import datetime
from datetime import date
import FileFinder.FileFinder as ff

#################################################   Global Variables    #############################################
macro_path = ff.get_path('macro_data.csv', 'Reference_Data')
sector_path = ff.get_path('sector_data.csv', 'Reference_Data')
factor_path = ff.get_path('factor_data.csv', 'Reference_Data')
available_macro_data = list(pd.read_csv(macro_path, index_col='Date').columns)
available_periods = ["full", "default", "uncertain_periods", "crashes", "normal_periods"]
string_today = datetime.datetime.strftime(date.today(), "%Y-%m-%d")

#####################################################################################################################



#################################################   FUCTIONS    #####################################################

def set_timeframe(df, timeframe):
    
    if isinstance(timeframe, tuple) or isinstance(timeframe, list):
        assert len(timeframe) == 2 or len(timeframe) == 1, "timeframe must have length of 1 or 2"
        if len(timeframe) == 2:
            df = df.loc[(df.index >= timeframe[0]) & (df.index <= timeframe[1])]
            return df
        else:
            df = df.loc[(df.index >= timeframe[0])]
            return df
    elif isinstance(timeframe, str):
        assert timeframe.lower() in available_periods, "if timeframe is str, must be string in variable 'available_periods'"
        if timeframe.lower() == 'default':
            df = df.loc[(df.index >= "2010-01-04") & (df.index <= string_today)]
            return df
        elif timeframe.lower() == 'full':
            return df
        else:
            return f"timeframe {timeframe} is not yet available.  In progress... "


def get_Dataset(factor_or_sector = None, desired_macro_data = None, timeframe = "default"):
    
    '''
    
    Inputs
        factor_or_sector (str): whether you want a dataset with factors or sectors as the obervation class
        
        desired_macro_data (list, str): which macro data series' you want to be the feature data for your dataset
        
        timeframe (tuple, list, str): either an interval of time ('Y-m-d', 'Y-m-d'), 
            or one of ["full", "default", "uncertain_periods", "crashes", "normal_periods"]
        
        
    Outputs
        Dataset with exact specifications
    
    '''
    
    assert isinstance(timeframe, (tuple, list, str)), "timeframe must be type list or tuple or str"
    assert isinstance(desired_macro_data, (list, str)), "desired_macro_data must be list or str"
    assert [i in available_macro_data for i in desired_macro_data], "A name you entered is not availble in macro_data. Please consult the 'available_macro_data' variable"
    
    macro_df = pd.read_csv(macro_path, index_col="Date")
    
    if 'fact' in factor_or_sector.lower():
        
        factor_df = pd.read_csv(factor_path, index_col="Date")
        requested_df = pd.concat([factor_df, macro_df[desired_macro_data]], axis = 1).dropna()
        return set_timeframe(requested_df, timeframe)

    elif 'sec' in factor_or_sector.lower():
        
        sector_df = pd.read_csv(sector_path, index_col="Date")
        requested_df = pd.concat([sector_df, macro_df[desired_macro_data]], axis = 1).dropna()
        return set_timeframe(requested_df, timeframe)

#####################################################################################################################


'''
print("Factor Data or Sector Data?, type either 'factor' or 'sector' ")
fac_or_sec = input()

print(f"Desired Macro Data, if any.  One or more of {available_macro_data}")
desired_macro = input()

print(f"Desired TimeFrame, can put interval of 'Y-M-D' or on of {available_periods}")
frame_of_reference = input()

get_Dataset(fac_or_sec, desired_macro, frame_of_reference)
'''