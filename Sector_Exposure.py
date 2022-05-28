import pandas as pd
import numpy as np
from collections import deque
#from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import fredapi as fa
import os
from datetime import datetime, timedelta, date
import json
from dateutil.relativedelta import relativedelta as delt


class CompareSectors(object):
    
    def __init__(self, df, look_back, forward, compare_against = 'Real Yield', benchmark = "SP-500"):
        
        self.look_back = look_back
        self.forward = forward
        self.df = df
        self.benchmark = benchmark
        
        self.real_yield = self.df[compare_against]
        '''
        sector_names = ['Energy', 'Materials', 'Industrials', 'Consumer Discretionary', 
                        'Consumer Staples', 'Health Care',
                        'Financials', 'Information Technology', 
                        'Telecommunication Services', 'Utilities', 'Real Estate']
        '''
        
        
        #sector_names = [i for i in df.columns if i != compare_against and i != self.benchmark]
        

        # CHANGED THE LENGTH FROM 5 TO 15
        gradient = pd.Series([(grad-self.real_yield[num])/15 for num, grad in enumerate(self.real_yield[15:])], index = [i for i in self.df.index[15:]])
        gradient_means = gradient.rolling(self.look_back, center=False).mean()
        gradient_std = gradient.rolling(self.look_back, center=False).std()

        real_yield_means = self.real_yield.rolling(self.look_back, center=False).mean()
        real_yield_std = self.real_yield.rolling(self.look_back, center=False).std()

        self.standardized_g = pd.Series((gradient[self.look_back:] - gradient_means[self.look_back:])/(gradient_std[self.look_back:]))
        self.standardized_df = pd.Series((self.real_yield[self.look_back:] - real_yield_means[self.look_back:])/(real_yield_std[self.look_back:]))
        
        self.raw_scores = self.standardized_g + self.standardized_df
        
        self.scores = self.raw_scores.copy()
        
        for num, s in enumerate(self.raw_scores):
            
            if round(s, 0) > 3.0 and s < 5.0:
                self.scores[num] = 4.0
            elif s > 5.0:
                self.scores[num] = 5.0
            elif round(s, 0) < -3.0 and s > -4.5:
                self.scores[num] = -4.0
            elif s < -4.5:
                self.scores[num] = -5.0
            else:
                self.scores[num] = round(s, 0)
            

        self.sectors = [i for i in df.columns if i != compare_against and i != self.benchmark]
        
        self.bucket_scores = { i : [] for i in self.sectors}
            
        self.df_formatted = self.df.copy()

        self.df_formatted['Date'] = [d for d in self.df.index]
        self.df_formatted.index = [i for i in range(len(self.df_formatted['Date']))]
        
        temp = self.get_total_returns(self.scores)
        temp['scores'] = self.scores
        
        self.total_returns = {}
        self.mean_returns = pd.DataFrame()
        
        for score in range(-5,6):
            
            self.total_returns[score] = temp.loc[temp.scores == score].drop(columns = ['scores'])
            self.mean_returns[score] = self.total_returns[score].mean()
            
        self.mean_returns[score] = self.mean_returns[score].T

    def get_total_returns(self, scores):
        
        frame = self.df.drop(columns = ['Real Yield', self.benchmark])
        
        ok = {i : [] for i in scores.index}
        
        for num, date in enumerate(frame.index):
            
            if (num+self.forward < len(self.df)):
                
                ok[date] = ( ((frame.iloc[num+self.forward] / frame.iloc[num]) - 1) - 
                            ((self.df[self.benchmark].iloc[num+self.forward] / self.df[self.benchmark].iloc[num]) - 1))
        
        temp = ok.copy()
        for t in temp:
            if len(temp[t]) == 0:
                ok.pop(t)
                
        ok = pd.DataFrame(ok).T
        ok.columns = self.sectors
        
        return ok

    
    def get_correlations(self, window = None):
        
        if not window:
            print('Please enter a value for window size')
        
        
        self.correlations = { i : { sec : deque() for sec in self.sectors} for i in range(-4,5)}
        self.correlations = { i : { sec : deque() for sec in self.sectors} for i in range(-4,5)}

        for i in self.scores:
            for date in self.scores[i]:
                
                df_index = self.df_formatted.index[self.df_formatted['Date'] == date][0]

                for sec in self.sectors:

                    if window > 0:
                        if (df_index+self.forward < len(self.df)) and (self.df[sec].iloc[df_index] != 0.0):
                        
                            self.correlations[i][sec].append(self.df[df_index:df_index+window].corr()[sec]['Real Yield'])
                    else:
                        if (df_index+self.forward < len(self.df)) and (self.df[sec].iloc[df_index] != 0.0):
                        
                            self.correlations[i][sec].append(self.df[df_index+window:df_index].corr()[sec]['Real Yield'])
                        
        self.correlation_means = { i : { sec : np.mean(self.correlations[i][sec]) for sec in self.sectors} for i in range(-4,5)}
        
        self.df_correlation_means = pd.DataFrame(self.correlation_means).T

        return self.df_correlation_means
    
    
    def show_return_distribution(self, sec = None, score = None):
        
        assert sec, score
        assert type(sec) == str and type(score) == int
        
        
        plt.figure(figsize=(12,8))
        self.total_returns[score][sec].hist()
        plt.title(f"{score} Signals for {sec} Sector", size = 20)
        plt.xlabel(f"{self.forward} Day Forward Returns", size = 14)
        plt.axvline(np.mean(self.total_returns[score][sec]), c = 'r')
        plt.tight_layout()
        plt.show()

