from statistics import correlation
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
import Useful_Functions as useful


class Rate_Exposures(object):
    
    def __init__(self, df, look_back, forward, compare_against = ['Real Yield'], benchmark = None):
        
        self.look_back = look_back
        self.forward = forward
        self.df = df
        self.benchmark = benchmark
        self.compare_against = compare_against
        
        self.scores = {}
        self.monthly_scores = {}
        
        for rate in compare_against:
            
            self.scores[rate] = self.get_scores(self.look_back, rate)
            self.monthly_scores[rate] = round(self.scores[rate].groupby(pd.PeriodIndex(self.scores[rate].index, freq="M")).mean(), 0)
        
        
        self.sectors = [i for i in df.columns if i not in compare_against and i != self.benchmark]
        self.bucket_scores = { i : [] for i in self.sectors}
        
        self.forward_returns_daily = self.get_total_returns()
        for rate in self.compare_against:
            self.forward_returns_daily[f"{rate}_scores"] = self.scores[rate]
        self.forward_returns_monthly = self.get_monthly_data(self.forward_returns_daily)
        for rate in self.compare_against:
            self.forward_returns_monthly[f"{rate}_scores"] = self.monthly_scores[rate]
        
        
        ### Signals
        self.combined_signals = pd.Series([(r,c) for r,c in zip(self.scores['Real Yield'].dropna(), self.scores['Yield Curve'].dropna())], 
                 index = self.scores['Yield Curve'].dropna().index)
        self.combined_signal_counts = self.combined_signals.value_counts().to_dict()
            
        
        self.removes = [f"{self.compare_against[0]}_scores", f"{self.compare_against[1]}_scores"]
        
        ### DAILY DATA ###
        self.total_returns = self.get_return_data(self.forward_returns_daily)
        self.mean_returns = pd.DataFrame({s : self.total_returns[s].mean() for s in self.total_returns})
        
        self.ry_total_returns = self.get_return_data(self.forward_returns_daily, "ry")
        self.ry_mean_returns = pd.DataFrame({s : self.ry_total_returns[s].mean() for s in self.ry_total_returns})
        self.yc_total_returns = self.get_return_data(self.forward_returns_daily, "yc")
        self.yc_mean_returns = pd.DataFrame({s : self.yc_total_returns[s].mean() for s in self.yc_total_returns})
        
        self.mean_returns = self.mean_returns.dropna(axis = 1).T
        self.mean_returns.index = pd.MultiIndex.from_tuples(list(self.mean_returns.index))
        self.ry_mean_returns = self.ry_mean_returns.T
        self.yc_mean_returns = self.yc_mean_returns.T
        ###
        
        ### MONTHLY DATA ###
        self.total_returns_monthly = self.get_return_data(self.forward_returns_monthly)
        self.mean_returns_monthly = pd.DataFrame({s : self.total_returns_monthly[s].mean() for s in self.total_returns_monthly})
        
        self.ry_total_returns_monthly = self.get_return_data(self.forward_returns_monthly, "ry")
        self.ry_mean_returns_monthly = pd.DataFrame({s : self.ry_total_returns_monthly[s].mean() for s in self.ry_total_returns_monthly})
        self.yc_total_returns_monthly = self.get_return_data(self.forward_returns_monthly, "yc")
        self.yc_mean_returns_monthly = pd.DataFrame({s : self.yc_total_returns_monthly[s].mean() for s in self.yc_total_returns_monthly})
        
        self.mean_returns_monthly = self.mean_returns_monthly.dropna(axis = 1).T
        self.mean_returns_monthly.index = pd.MultiIndex.from_tuples(list(self.mean_returns_monthly.index))
        self.ry_mean_returns_monthly = self.ry_mean_returns_monthly.T
        self.yc_mean_returns_monthly = self.yc_mean_returns_monthly.T
        ###
        
        
    
    
    def get_return_data(self, returns_data, rate = "both"):
        
        temp = {}
        
        for score in range(-6,7):
            
            if "real" in rate.lower() or "ry" in rate.lower():
                consider = f"{self.compare_against[0]}_scores"
            elif "curve" in rate.lower() or "yc" in rate.lower():
                consider = f"{self.compare_against[1]}_scores"
            if rate.lower() != "both":
                # .drop(columns = self.compare_against)
                temp[score] = returns_data.loc[returns_data[consider] == score].drop(columns = self.removes)
        
            if rate.lower() == "both":
                
                for score_2 in range(-6,7):
                    # .drop(columns = self.compare_against)
                    data = returns_data.loc[(returns_data[f"{self.compare_against[1]}_scores"] == score_2) & 
                                            (returns_data[f"{self.compare_against[0]}_scores"] == score)].drop(columns = [f"{self.compare_against[0]}_scores", 
                                                                                                                          f"{self.compare_against[1]}_scores"])
                    
                    if len(data) > 0:
                        temp[(score, score_2)] = data
        
        return temp
    
        
    def get_scores(self, lookback, target):
    
        target_rate = self.df[target]

        gradient = pd.Series([(grad-target_rate[num])/5 for num, grad in enumerate(target_rate[5:])], index = [i for i in self.df.index[5:]])
        gradient_means = gradient.rolling(lookback, center=False).mean()
        gradient_std = gradient.rolling(lookback, center=False).std()

        target_rate_means = target_rate.rolling(lookback, center=False).mean()
        target_rate_std = target_rate.rolling(lookback, center=False).std()

        standardized_g = pd.Series((gradient[lookback:] - gradient_means[lookback:])/(gradient_std[lookback:]))
        standardized_df = pd.Series((target_rate[lookback:] - target_rate_means[lookback:])/(target_rate_std[lookback:]))
        
        raw_scores = standardized_g + standardized_df
        
        scores = round(raw_scores, 0)

        return scores.dropna()
    
    def get_monthly_data(self, data):
        
        return data.groupby(pd.PeriodIndex(data.index, freq="M"))[[i for i in data.columns]].mean()
                

    def get_total_returns(self):
        
        rets = {}
        
        for num, date in enumerate(self.df.index):
            
            if (num+self.forward < len(self.df)):
                
                if self.benchmark:
                    
                    rets[date] = (((self.df[list(self.sectors)].iloc[num+self.forward] / self.df[list(self.sectors)].iloc[num]) - 1) - 
                            ((self.df[self.benchmark].iloc[num+self.forward] / self.df[self.benchmark].iloc[num]) - 1))
                
                else:
                    
                    rets[date] = (self.df[list(self.sectors)].iloc[num+self.forward] / self.df[list(self.sectors)].iloc[num]) - 1
                    
        
        temp = rets.copy()
        for t in temp:
            if len(temp[t]) == 0:
                rets.pop(t)
        
        rets = pd.DataFrame(rets).T
        rets.columns = self.sectors
        
        return rets

    
    def get_betas_x_days_after_signal(self, window = None, rate = 'both', full = False):
        
        '''
        
        Full being False inplies that the data is showing correlation Means
        
        '''
        
        assert window, 'Please enter a value for window size'
        assert isinstance(window, int), "window must be an int"
        assert window > 0, "please enter a positive integer for window"
        assert isinstance(full, bool), "full must be a boolean"
        
        betas = {}
        
        drops = [col for col in self.df.columns if col == self.benchmark or col in self.compare_against]
        
        consider_df = self.df.pct_change()

        for num, date in enumerate(self.df.index):
            
            if num > 0 and (num+window < len(self.df)):
                
                # (self.df[num:num+window].corr()[self.benchmark])
                betas[date] = {}
                
                for sec in self.sectors:
                    betas[date][sec] = useful.beta_asset_to_index(consider_df[[sec, self.benchmark]][num:num+window].values)
                    
        # .drop(columns = [self.benchmark])
        betas = pd.DataFrame(betas).T
        
        
        for num, score in enumerate(self.scores):
            
            betas[f"{self.compare_against[num]}_scores"] = self.scores[score]
        
        betas = self.get_return_data(betas.dropna(), rate)

        if not full:
            return pd.DataFrame({score : betas[score].mean() for score in betas}).T
        else:
            return betas
    
    def get_correlation_x_days_after_signal(self, window = None, rate = 'both', full = False):
        
        '''
        
        Full being False inplies that the data is showing correlation Means
        
        '''
        
        assert window, 'Please enter a value for window size'
        assert isinstance(window, int), "window must be an int"
        assert window > 0, "please enter a positive integer for window"
        assert isinstance(full, bool), "full must be a boolean"
        
        correlations = {}
        
        drops = [col for col in self.df.columns if col == self.benchmark or col in self.compare_against]
        
        consider_df = self.df.pct_change()

        for num, date in enumerate(self.df.index):
            
            if num > 0 and (num+window < len(self.df)):
                
                # (self.df[num:num+window].corr()[self.benchmark])
                correlations[date] = consider_df[num:num+window].corr()[self.benchmark].drop(index = drops)
                
                #for sec in self.sectors:
                    #correlations[date][sec] = useful.beta_asset_to_index(consider_df[[sec, self.benchmark]][num:num+window].values)
                    
        # .drop(columns = [self.benchmark])
        correlations = pd.DataFrame(correlations).T
        
        for num, score in enumerate(self.scores):
            
            correlations[f"{self.compare_against[num]}_scores"] = self.scores[score]
        
        correlations = self.get_return_data(correlations.dropna(), rate)

        if not full:
            # .drop(columns = drops)
            return pd.DataFrame({score : correlations[score].mean() for score in correlations}).T
        else:
            return correlations
    
    
    def show_return_distribution(self, monthly = True, sec = None, score = None, rate = "Both"):
        
        assert sec, score
        assert type(sec) == str and (type(score) == int or type(score) == tuple)
        
        plt.figure(figsize=(12,8))
        
        if type(score) == tuple:
            if monthly:
                data = self.total_returns_monthly
            else:
                data = self.total_returns
            data[score][sec].hist()
            plt.title(f"{(score[0], score[1])} Signals for {sec} Sector", size = 20)
            plt.xlabel(f"{self.forward} Day Forward Returns", size = 14)
            plt.axvline(np.mean(data[score][sec]), c = 'r')
            
        else:
            
            if "real" in rate:
                if monthly:
                    data = self.ry_total_returns_monthly
                else:
                    data = self.ry_total_returns
                data[score][sec].hist()
                plt.title(f"{score} Signals for {sec} Sector", size = 20)
                plt.xlabel(f"{self.forward} Day Forward Returns", size = 14)
                plt.axvline(np.mean(data[score][sec]), c = 'r')
                
            elif "curve" in rate:
                if monthly:
                    data = self.yc_total_returns_monthly
                else:
                    data = self.yc_total_returns
                data[score][sec].hist()
                plt.title(f"{score} Signals for {sec} Sector", size = 20)
                plt.xlabel(f"{self.forward} Day Forward Returns", size = 14)
                plt.axvline(np.mean(data[score][sec]), c = 'r')
        
        plt.tight_layout()
        plt.show()
        
        

    def signal_dates_and_returns(self, monthly = True, sector = None, scores = None, rate = "both"):
        
        assert self.total_returns, self.total_returns_monthly
        assert self.ry_total_returns, self.ry_total_returns_monthly
        assert self.yc_total_returns, self.yc_total_returns_monthly
        
        assert sector in self.sectors
        assert scores
        
        if rate.lower() == "both" or isinstance(scores, tuple):
            
            if monthly:
                return self.total_returns_monthly[scores][sector].sort_values()
            else:
                return self.total_returns[scores][sector].sort_values()
        
        elif ("real" in rate.lower() or "ry" in rate.lower()) and isinstance(scores, int):
            
            if monthly:
                return self.ry_total_returns_monthly[scores][sector].sort_values()
            else:
                return self.ry_total_returns[scores][sector].sort_values()
        
        elif ("curve" in rate.lower() or "yc" in rate.lower()) and isinstance(scores, int):
            
            if monthly:
                return self.yc_total_returns_monthly[scores][sector].sort_values()
            else:
                return self.yc_total_returns[scores][sector].sort_values()