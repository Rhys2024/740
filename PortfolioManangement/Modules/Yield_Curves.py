from copy import deepcopy
import pandas as pd
from Sector_Exposure import Rate_Exposures
import os
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


class YieldCurve(object):
    
    
    def __init__(self, df, yield_curve, long, short, forward = 252, index = None, max_forward = None):
        
        # Basic Allocation of Input Variables
        self.df = df
        self.yield_curve = yield_curve
        self.long = long
        self.short = short
        self.forward = forward
        self.max = max_forward
        self.dex = index
        
        # Filter Data Into Segments for Obervation
        self.sector_names = ['Energy', 'Materials', 'Industrials', 'Consumer Discretionary', 
                        'Consumer Staples', 'Health Care',
                        'Financials', 'Information Technology', 
                        'Telecommunication Services', 'Utilities', 'Real Estate']
        self.patterns = ['Bull Flattening', 'Bear Flattening', 'Bull Steepening', 'Bear Steepening']
        
        # Behavior of 10-Year vs 2-Year
        self.ten_change = pd.Series([(chng/self.df[self.long][num])-1 for num, chng in enumerate(self.df[self.long][30:])], index = [i for i in df[self.long][30:].index])
        self.two_change = pd.Series([(chng/self.df[self.short][num])-1 for num, chng in enumerate(self.df[self.short][30:])], index = [i for i in df[self.short][30:].index])
        self.ten_change_mean = self.ten_change.rolling(60).mean()[60:]
        self.two_change_mean = self.two_change.rolling(60).mean()[60:]

        self.pattern_df = self.get_patterns()
        self.lengths_df = self.get_pattern_lengths()
        
        self.df['Pattern'] = self.add_patterns()
        self.df['Inverted'] = self.is_inverted()
        self.df['Curve Behavior'] = self.combine()
        self.df['Pattern-Inversion'] = self.combo
        
        if self.dex:
            self.excess_returns = self.get_returns(excess = True)
        self.returns = self.get_returns(excess = False)
        
        if self.dex:
            self.mean_excess_returns = self.get_mean_returns(excess=True)
        self.mean_returns = self.get_mean_returns(excess=False)
        
        if self.dex:
            self.excess_pattern_periods = self.describe_periods(excess=True)
        self.pattern_periods = self.describe_periods(excess=False)
        
        if self.dex:
            self.depth_excess_pattern_return = self.in_depth(excess=True)
        self.depth_pattern_return = self.in_depth(excess=False)
        

    def is_inverted(self):
        
        return np.select([self.df[self.yield_curve] >= 0.0, self.df[self.yield_curve] < 0.0], [0,1])
        

    def length_of_pattern(self, series):
    
        count = 0
        
        while (series[count] != 0):
            
            count += 1
            
            if (count >= len(series)):
                return count
            
        return count
    
    
    def get_patterns(self):
        
        # , self.df[self.yield_curve]
        ten_and_two = pd.DataFrame([self.ten_change_mean, self.two_change_mean]).T
        
        # , self.yield_curve
        ten_and_two.columns = ['10-Year Change', '2-Year Change']

        ten_and_two['Bull Flattening'] = 0
        ten_and_two['Bear Flattening'] = 0
        ten_and_two['Bull Steepening'] = 0
        ten_and_two['Bear Steepening'] = 0
        #ten_and_two['Inverted Curve'] = 0
        ten_and_two['Pattern'] = 'None'

        for date in ten_and_two.index:
            
            if ten_and_two.loc[date, '10-Year Change'] > 0 and ten_and_two.loc[date, '2-Year Change'] > 0:
            
                if ten_and_two.loc[date, '10-Year Change'] > ten_and_two.loc[date, '2-Year Change']:
                    
                    ten_and_two.loc[date, 'Bear Steepening'] = 1
                    ten_and_two.loc[date, 'Pattern'] = 'Bear Steepening'
                
                elif ten_and_two.loc[date, '10-Year Change'] < ten_and_two.loc[date, '2-Year Change']:
                    
                    ten_and_two.loc[date, 'Bear Flattening'] = 1
                    ten_and_two.loc[date, 'Pattern'] = 'Bear Flattening'
                    
            elif ten_and_two.loc[date, '10-Year Change'] < 0 and ten_and_two.loc[date, '2-Year Change'] < 0:
                
                if ten_and_two.loc[date, '10-Year Change'] > ten_and_two.loc[date, '2-Year Change']:
                    
                    ten_and_two.loc[date, 'Bull Steepening'] = 1
                    ten_and_two.loc[date, 'Pattern'] = 'Bull Steepening'
                
                elif ten_and_two.loc[date, '10-Year Change'] < ten_and_two.loc[date, '2-Year Change']:
                    
                    ten_and_two.loc[date, 'Bull Flattening'] = 1
                    ten_and_two.loc[date, 'Pattern'] = 'Bull Flattening'
                    
        
            #if ten_and_two.loc[date, self.yield_curve] < 0.0:
                #ten_and_two.loc[date, 'Inverted Curve'] = 1
                #ten_and_two.loc[date, 'Pattern'] = 'Inverted Curve'
                    
                    
        return ten_and_two
    
    
    def get_pattern_lengths(self):

        lengths = {pat : {} for pat in self.patterns}

        for p in self.patterns:

            for num, i in enumerate(self.pattern_df[p]):
                
                if num > 0 and i == 1 and self.pattern_df[p][num-1] != 1 and num + 1 < len(self.pattern_df):
                    
                    lengths[p][self.pattern_df.index[num]] = self.length_of_pattern(self.pattern_df[p][num:])
                    
                    
        return pd.DataFrame(lengths)
    
    
    def add_patterns(self):

        count = 0
        general_pattern = []
        
        self.lengths_df['Yield Curve'] = self.df['Yield Curve']
        self.df['Types'] = self.pattern_df['Pattern'].fillna('None')
        #self.df['Types'] = self.df['Types']
        self.df['Lengths'] = 0
        
        lengths = self.lengths_df.fillna(0.0)
        self.shortest_pattern = min([max(lengths[i]) for i in lengths if i != 'Yield Curve'])


        for date in self.lengths_df.index:
            
            for p in self.patterns:
                
                    
                if self.lengths_df.loc[date, p] > (self.shortest_pattern - 10):
                    
                    self.df.loc[date, 'Lengths'] = self.lengths_df.loc[date, p]

        while count < len(self.df.index):
            
            if self.df.loc[self.df.index[count], 'Lengths'] > 0:
                
                for i in range(int(self.df.loc[self.df.index[count], 'Lengths'])):
                    
                    #if self.df.loc[self.df.index[count+i], 'Yield Curve'] < 0.05:
                        #general_pattern.append('Inverted Curve')
                    #else:
                    general_pattern.append(self.df.loc[self.df.index[count], 'Types'])
                
                count += int(self.df.loc[self.df.index[count], 'Lengths'])
                
            else:
                
                #if (count > 5) and (self.df.loc[self.df.index[count], 'Yield Curve'] != 0) and (self.df.loc[self.df.index[count], 'Yield Curve'] < 0.16) and ((self.df.loc[self.df.index[count], 'Yield Curve'] - self.df.loc[self.df.index[count - 25], 'Yield Curve']) < 0.0):
                    #general_pattern.append('Inverted Curve')
                #if self.df.loc[self.df.index[count], 'Yield Curve'] < 0.05:
                    #general_pattern.append('Inverted Curve')
                #else:
                general_pattern.append('None')
                    
                count += 1
                
                
        general_pattern = pd.Series(general_pattern, index = self.df.index)

        return general_pattern
    
    
    def get_returns(self, excess = True):
        
        pats = ['Bull Flattening', 'Bear Flattening', 'Bull Steepening', 'Bear Steepening', 'Inverted Curve', 'None']
        
        leave_out = ['Types', 'Lengths', 'Pattern', '2-Year', '10-Year', 'Yield Curve', 'Real Yield', 'Inverted', 'Pattern-Inversion', 'Curve Behavior']
        
        all_secs = [col for col in self.df.columns if col not in leave_out]
        
        rets = {p : {sec : {'Date' : [], 'Forward Return': []} for sec in all_secs} for p in pats}
        

        for p in pats:
            
            if p != 'Inverted Curve':
                
                for num, date in enumerate(self.df.index):
                    
                    if self.max:
                    
                        if self.df.loc[date, 'Pattern'] == p and num + self.max < len(self.df):
                                
                            for sec in all_secs:
                                    
                                rets[p][sec]['Date'].append(date)
                                rets[p][sec]['Forward Return'].append((self.df[sec][num+252] / self.df[sec][num]) - 1)
                                    
                            if self.df.loc[date, 'Inverted']:
                                
                                for sec in all_secs:
                                    
                                    rets['Inverted Curve'][sec]['Date'].append(date)
                                    rets['Inverted Curve'][sec]['Forward Return'].append((self.df[sec][num+252] / self.df[sec][num]) - 1)
                    
                    else:
                        
                        if self.df.loc[date, 'Pattern'] == p and num + self.forward < len(self.df):
                                
                            for sec in all_secs:
                                    
                                rets[p][sec]['Date'].append(date)
                                rets[p][sec]['Forward Return'].append((self.df[sec][num+252] / self.df[sec][num]) - 1)
                                
                            if self.df.loc[date, 'Inverted']:
                                
                                for sec in all_secs:
                                    
                                    rets['Inverted Curve'][sec]['Date'].append(date)
                                    rets['Inverted Curve'][sec]['Forward Return'].append((self.df[sec][num+252] / self.df[sec][num]) - 1)
                                    
  
        for p in rets:
            for sec in rets[p]:
                
                if excess:
                        
                    if sec != self.dex:
                
                        rets[p][sec] = pd.Series(rets[p][sec]['Forward Return'], index = rets[p][sec]['Date']) - pd.Series(rets[p][self.dex]['Forward Return'], 
                                                                                                                    index = rets[p][self.dex]['Date'])
                    
                else:
                        
                    rets[p][sec] = pd.Series(rets[p][sec]['Forward Return'], index = rets[p][sec]['Date'])
                        
            
        for p in rets:
            
            if not excess:
                rets[p] = pd.DataFrame(rets[p])
                
            else:
                rets[p].pop(list(rets[p].keys())[-1])
                rets[p] = pd.DataFrame(rets[p])
                
        
        return rets
    
    
    
    def get_mean_returns(self, excess = True):
    
        pats = ['Bull Flattening', 'Bear Flattening', 'Bull Steepening', 'Bear Steepening', 'Inverted Curve', 'None']
            
        leave_out = ['Types', 'Lengths', 'Pattern', '2-Year', '10-Year', 'Yield Curve', 'Real Yield', self.dex, 'Inverted', 'Pattern-Inversion', 'Curve Behavior']
            
        all_secs = [col for col in self.df.columns if col not in leave_out]
        
        mean_rets = {p : {sec : 0.0 for sec in all_secs} for p in pats}
        
        if excess:
            feature_data = self.excess_returns
            
        else:
            
            feature_data = self.returns
            
            
        for pat in feature_data:
            
            for sec in feature_data[pat]:
                
                mean_rets[pat][sec] = np.mean(feature_data[pat][sec])
        
        return pd.DataFrame(mean_rets).T
    
    def combine(self):
        
        self.combo = [(p,i) for p,i in zip(self.df['Pattern'], self.df['Inverted'])]
 
        pattern_inversion = []
        
        for pi in self.combo:
            
            if not pi[1]:
                pattern_inversion.append(pi[0])
            else:
                pattern_inversion.append('Inverted Curve')
                
        return pattern_inversion
        
    def visualize(self):
        
        fig = px.scatter(self.df, x = self.df.index, y = self.yield_curve, color = 'Curve Behavior',labels={
                     #"sepal_length": "Sepal Length (cm)",
                     "Yield Curve": f"{self.long} to {self.short} Yield Curve",
                     "Patterns": "Bull/Bear Patterns and Inversion"
                 }, title = f'Bull/Bear Patterns and Inversions on The {self.long} Bonds to {self.short} Bonds {self.yield_curve}')
        
        fig.show()
        
        
        
    def describe_periods(self, excess = True):
            
        format = []
        pees = {p : 0 for p in self.patterns}
        
        if excess:
            feat_data = self.excess_returns
        else:
            feat_data = self.returns

        for num, (p, l) in enumerate(zip(self.df['Pattern'], self.df['Lengths'])):
            
            if l > 0:
                pees[p] += 1
                format.append([p, pees[p], l, self.df.index[num]]) 

        self.periods = pd.DataFrame(format, columns = ['Pattern', 'Number', 'Duration', 'Start Date'])
        self.periods.index = self.periods['Number']
        self.periods['End Date'] = self.periods['Start Date'].shift(-1)

        periods_returns = {p: { sec : {num+1 : [] for num, date in enumerate(self.periods['Start Date'].loc[self.periods['Pattern'] == p]) if date + pd.Timedelta(days = 252) < self.df.index[-1]} 
                            for sec in self.sector_names}
                        for p in list(self.periods['Pattern'].unique())}

        for p in list(self.periods['Pattern'].unique()):
            
            cop = feat_data[p].copy(deep = True)
                        
            cop['Date'] = cop.index
                        
            cop.index = [i for i in range(len(cop))]
            
            for sec in self.sector_names:
            
                for num, start_date in enumerate(self.periods['Start Date'].loc[self.periods['Pattern'] == p]):
                        
                    if start_date + pd.Timedelta(days = 252) < self.df.index[-1]:
                            
                        duration = self.periods['Duration'].loc[self.periods['Start Date'] == start_date]
                            
                        dex = cop[sec].loc[cop['Date'] == start_date].index[0]
                            
                        for r in cop[sec][dex:dex+int(duration)+1]:
                                
                            periods_returns[p][sec][num+1].append(r)
                    
                    
                        periods_returns[p][sec][num+1] = pd.Series(periods_returns[p][sec][num+1], index = cop['Date'][dex:dex+int(duration)+1])
                        
        return periods_returns
                        
          
          
    def in_depth(self, excess = True):
        
        if excess:
            feat_data = self.excess_pattern_periods
        else:
            feat_data = self.pattern_periods
                      
        in_depth_return = {p : {sec : {num : {"Start Date" : feat_data[p][sec][num].index[0].strftime("%Y-%m-%d"),
                                              "End Date" : (feat_data[p][sec][num].index[0] + timedelta( days = len(feat_data[p][sec][num]))).strftime("%Y-%m-%d"),
                                        "Duration" : len(feat_data[p][sec][num]),
                                        "Returns" : np.mean(feat_data[p][sec][num])}
                                for num in feat_data[p][sec]} 
                            for sec in feat_data[p]} for p in feat_data}
        
        pattern_returns = {p: pd.DataFrame() for p in feat_data}

        for p in pattern_returns:
            
            in_depth_df = pd.DataFrame(in_depth_return[p]['Energy']).T
            in_depth_df.columns = ['Start Date', 'End Date', 'Duration', 'Energy']

            for sec in self.sector_names:
                    
                    if sec != "Energy":
                            
                            in_depth_df[sec] = [in_depth_return[p][sec][num]['Returns'] for num in feat_data[p][sec]]
            
            pattern_returns[p] = in_depth_df
            
        return pattern_returns
    
    
    def to_excel(self, filename, frame):

        path = os.path.realpath(f"{filename}.xlsx")
        writer = pd.ExcelWriter(path, engine='xlsxwriter')
        
        if isinstance(frame, dict):
            
            for key in frame:
                
                frame[key].to_excel(writer, sheet_name=f'{key}')
                
        else:
            
            frame.to_excel(writer, sheet_name=f'{filename}')
        
        writer.save()