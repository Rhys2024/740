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


class Rate_Exposures(object):
    
    def __init__(self, df, look_back, forward, compare_against = ['Real Yield'], benchmark = "SP-500"):
        
        self.look_back = look_back
        self.forward = forward
        self.df = df
        self.benchmark = benchmark
        self.compare_against = compare_against
        
        # FIX THISSSSSSSSSSSSSSSSSSS
        #self.real_yield = self.df[self.compare_against]
        
        '''
        sector_names = ['Energy', 'Materials', 'Industrials', 'Consumer Discretionary', 
                        'Consumer Staples', 'Health Care',
                        'Financials', 'Information Technology', 
                        'Telecommunication Services', 'Utilities', 'Real Estate']
        '''
        
        #self.scores = self.get_scores()
        
        #sector_names = [i for i in df.columns if i != compare_against and i != self.benchmark]
        
        '''
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
        '''
        
        self.scores = {}
        
        for rate in compare_against:
            
            self.scores[rate] = self.get_scores(self.look_back, rate)

        self.sectors = [i for i in df.columns if i != compare_against and i != self.benchmark]
        
        self.bucket_scores = { i : [] for i in self.sectors}
        
        ### Fix Correlations and Delete Formatted DF ###
        self.df_formatted = self.df.copy()
        self.df_formatted['Date'] = [d for d in self.df.index]
        self.df_formatted.index = [i for i in range(len(self.df_formatted['Date']))]
        
        temp = self.get_total_returns()
        
        for rate in self.compare_against:
            temp[f"{rate}_scores"] = self.scores[rate]
        
        self.total_returns = {}
        self.mean_returns = pd.DataFrame()
        
        self.ry_total_returns = {}
        self.ry_mean_returns = pd.DataFrame()
        self.yc_total_returns = {}
        self.yc_mean_returns = pd.DataFrame()
        
        removes = [f"{self.compare_against[0]}_scores", f"{self.compare_against[1]}_scores"]
        
        for score in range(-5,6):
            
            # [f"{self.compare_against[0]}_scores"]
            self.ry_total_returns[score] = temp.loc[temp[f"{self.compare_against[0]}_scores"] == score].drop(columns = removes).drop(columns = self.compare_against)
            self.ry_mean_returns[score] = self.ry_total_returns[score].mean()
            
            self.yc_total_returns[score] = temp.loc[temp[f"{self.compare_against[1]}_scores"] == score].drop(columns = removes).drop(columns = self.compare_against)
            self.yc_mean_returns[score] = self.yc_total_returns[score].mean()
            
            for score_2 in range(-5,6):
                
                self.total_returns[(score, score_2)] = temp.loc[(temp[f"{self.compare_against[1]}_scores"] == score_2) & (temp[f"{self.compare_against[0]}_scores"] == score)].drop(columns = [f"{self.compare_against[0]}_scores", f"{self.compare_against[1]}_scores"]).drop(columns = self.compare_against)
                self.mean_returns[(score, score_2)] = self.total_returns[(score, score_2)].mean()
        
        # [(score, score_2)]
        self.mean_returns = self.mean_returns.dropna(axis = 1).T
        self.mean_returns.index = pd.MultiIndex.from_tuples(list(self.mean_returns.index))
        self.ry_mean_returns = self.ry_mean_returns.T
        self.yc_mean_returns = self.yc_mean_returns.T
        
        
    def get_scores(self, lookback, target):
    
        target_rate = self.df[target]

        #print(target_rate)
        gradient = pd.Series([(grad-target_rate[num])/15 for num, grad in enumerate(target_rate[15:])], index = [i for i in self.df.index[15:]])
        gradient_means = gradient.rolling(lookback, center=False).mean()
        gradient_std = gradient.rolling(lookback, center=False).std()

        target_rate_means = target_rate.rolling(lookback, center=False).mean()
        target_rate_std = target_rate.rolling(lookback, center=False).std()

        standardized_g = pd.Series((gradient[lookback:] - gradient_means[lookback:])/(gradient_std[lookback:]))
        standardized_df = pd.Series((target_rate[lookback:] - target_rate_means[lookback:])/(target_rate_std[lookback:]))
        
        raw_scores = standardized_g + standardized_df
        
        scores = round(raw_scores, 0)
        
        #self.scores = self.raw_scores.copy()

        return scores
                

    def get_total_returns(self):
        
        #take_out = self.compare_against.copy()
        #take_out.append(self.benchmark)
        
        #frame = self.df[[list(self.sectors)]]
        
        ok = {}
        
        for num, date in enumerate(self.df.index):
            
            if (num+self.forward < len(self.df)):
                
                ok[date] = (((self.df[list(self.sectors)].iloc[num+self.forward] / self.df[list(self.sectors)].iloc[num]) - 1) - 
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
    
    
    def show_return_distribution(self, sec = None, score = None, rate = "Both"):
        
        assert sec, score
        assert type(sec) == str and (type(score) == int or type(score) == tuple)
        
        
        plt.figure(figsize=(12,8))
        if type(score) == tuple:
            self.total_returns[score][sec].hist()
            plt.title(f"{(score[0], score[1])} Signals for {sec} Sector", size = 20)
            plt.xlabel(f"{self.forward} Day Forward Returns", size = 14)
            plt.axvline(np.mean(self.total_returns[score][sec]), c = 'r')
            
        else:
            if "real" in rate:
                self.ry_total_returns[score][sec].hist()
                plt.title(f"{score} Signals for {sec} Sector", size = 20)
                plt.xlabel(f"{self.forward} Day Forward Returns", size = 14)
                plt.axvline(np.mean(self.ry_total_returns[score][sec]), c = 'r')
                
            elif "curve" in rate:
                self.yc_total_returns[score][sec].hist()
                plt.title(f"{score} Signals for {sec} Sector", size = 20)
                plt.xlabel(f"{self.forward} Day Forward Returns", size = 14)
                plt.axvline(np.mean(self.yc_total_returns[score][sec]), c = 'r')
        
        plt.tight_layout()
        plt.show()
        #else:
            #plt.figure(figsize=(12,8))
            #self.total_returns[score][sec].hist()
            #plt.title(f"{score} Signals for {sec} Sector", size = 20)
            #plt.xlabel(f"{self.forward} Day Forward Returns", size = 14)
            #plt.axvline(np.mean(self.total_returns[score][sec]), c = 'r')
            #plt.tight_layout()
            #plt.show()
