import pandas as pd
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats.mstats import normaltest

class Exposure(object):
    
    def __init__(self, df, look_back, forward, compare_against = ['Real Yield'], benchmark = None, ):
        
        self.look_back = look_back
        self.forward = forward
        self.df = df
        self.benchmark = benchmark
        self.compare_against = compare_against
        
        self.scores = {}
        self.raw_scores = {}
        self.monthly_scores = {}
        self.raw_scores_monthly = {}
        
        for rate in compare_against:
            
            score_data = self.get_scores(self.look_back, rate)
            self.scores[rate] = score_data[0]
            self.raw_scores[rate] = score_data[1]
            #self.monthly_scores[rate] = round(self.scores[rate].groupby(pd.PeriodIndex(self.scores[rate].index, freq="M")).mean(), 0)
            self.monthly_scores[rate] = round(self.scores[rate].rolling(21).mean(), 0).dropna()
            self.raw_scores_monthly[rate] = round(self.raw_scores[rate].rolling(21).mean(), 3).dropna()

        self.sectors = [i for i in df.columns if i not in compare_against and i != self.benchmark]
        self.bucket_scores = { i : [] for i in self.sectors}
        
        self.forward_returns_daily = self.get_total_returns()
        self.forward_returns_daily_raw = self.forward_returns_daily.copy()
        for rate in self.compare_against:
            self.forward_returns_daily[f"{rate}_scores"] = self.scores[rate]
            self.forward_returns_daily_raw[f"{rate}_scores"] = self.raw_scores[rate]
        self.forward_returns_daily = self.forward_returns_daily.dropna()
        self.forward_returns_daily_raw = self.forward_returns_daily_raw.dropna()
        #self.forward_returns_monthly = self.get_monthly_data(self.forward_returns_daily)
        self.forward_returns_monthly = self.forward_returns_daily.rolling(21).mean().dropna()
        self.forward_returns_monthly_raw = self.forward_returns_monthly.copy()
        for rate in self.compare_against:
            self.forward_returns_monthly[f"{rate}_scores"] = self.monthly_scores[rate]
            self.forward_returns_monthly_raw[f"{rate}_scores"] = self.raw_scores_monthly[rate]
            
        self.forward_returns_monthly = self.forward_returns_monthly.dropna()
        self.forward_returns_monthly_raw = self.forward_returns_monthly_raw.dropna()
        
        self.removes = [f"{rate}_scores" for rate in self.compare_against]
        
        self.total_returns = {}
        self.total_returns_monthly = {}
        self.mean_returns = {}
        self.mean_returns_monthly = {}
        self.signal_counts = {}
        
        for rate in self.scores:
            
            self.total_returns[rate] = self.get_return_data(self.forward_returns_daily, rate)
            self.total_returns_monthly[rate] = self.get_return_data(self.forward_returns_monthly, rate, False)
            self.signal_counts[rate] = self.scores[rate].value_counts().to_dict()
            self.mean_returns[rate] = pd.DataFrame({s : self.total_returns[rate][s].mean() for s in self.total_returns[rate]}).dropna().T
            self.mean_returns_monthly[rate] = pd.DataFrame({s : self.total_returns_monthly[rate][s].mean() for s in self.total_returns_monthly[rate]}).dropna().T
            
            #is_normal = []
            
            #for s in self.total_returns[rate]:
                
                #try:
                    #is_normal.append(normaltest(self.total_returns[rate][s])[1] < .05)
                #except:
                    #is_normal.append(np.array(['Insufficient Data' for i in range(len(self.total_returns[rate][s].columns))]))

        #self.is_normal_dist = pd.DataFrame(is_normal, columns = self.mean_returns["Real Yield"].columns, index = self.mean_returns[rate].index)
        
        
        ### Signals
        if len(self.compare_against) == 2:
            
            self.first_two_rates_combined = pd.Series([(r,c) for r,c in zip(self.scores[self.compare_against[0]].dropna(), self.scores[self.compare_against[1]].dropna())], 
                        index = self.scores[self.compare_against[0]].dropna().index)
            self.signal_counts_first_two_rates_combined = self.first_two_rates_combined.value_counts().to_dict()
    
            combo = self.get_combo_data(self.compare_against)
            combo_monthly = self.get_combo_data(self.compare_against, False)
            self.total_return_combo = combo['total_combo_returns']
            self.mean_return_combo = combo['mean_combo_returns']
            self.total_return_combo_monthly = combo_monthly['total_combo_returns']
            self.mean_return_combo_monthly = combo_monthly['mean_combo_returns']
        
    
    def __get_combo_scores(self, combo_cols, forward_ret_df):
        
        return [tuple(i) for i in forward_ret_df[list(combo_cols.values())].values]
    
    
    def get_combo_data(self, combo = None, daily = True):
    
        assert isinstance(combo, list), "combo must me a list of comparable rates"
        
        if daily:
            forward_ret_copy = self.forward_returns_daily.copy()
        else:
            forward_ret_copy = self.forward_returns_monthly.copy()
        
        combo_cols = {rate : f"{rate}_scores" for rate in combo}
        past_combinations = pd.Series(self.__get_combo_scores(combo_cols, forward_ret_copy), index = forward_ret_copy.index)
        
        # forward_ret_copy = forward_ret_copy.iloc[:,:-(len(self.compare_against))]
        forward_ret_copy = forward_ret_copy[self.sectors]
        forward_ret_copy['combos'] = past_combinations
        unique_combos = past_combinations.sort_values().unique()
        combo_rets = {}
        mean_combo_rets = {}
        
        for un in unique_combos:
            #combo_rets[un] = forward_ret_copy.iloc[:,:-1].loc[ forward_ret_copy.combos == un]
            combo_rets[un] = forward_ret_copy[self.sectors].loc[forward_ret_copy.combos == un]
            mean_combo_rets[un] = combo_rets[un].mean()
        
        mean_combo_rets = pd.DataFrame(mean_combo_rets).T
        
        return {"total_combo_returns" : combo_rets, "mean_combo_returns" : mean_combo_rets}



    def get_return_data(self, returns_data, rate = None, daily = True):
        
        if daily:
            score_range = [int(i) for i in self.scores[rate].dropna()[:-self.forward].sort_values().unique()]
        else:
            score_range = [int(i) for i in self.monthly_scores[rate].dropna()[:-self.forward].sort_values().unique()]
            
        temp = {}
        
        for score in score_range:
            
            consider = f"{rate}_scores"
            
            temp[score] = returns_data.loc[returns_data[consider] == score].drop(columns = self.removes)
                
        return temp
    
        
    def get_scores(self, lookback, target):
        
        gradient = self.df[target].diff(10)

        # min_periods = 252, window = len(ry)-1).mean()
        length = len(self.df[target])
        regular_scores = (self.df[target] - self.df[target].rolling(lookback, min_periods = 60).mean()) / self.df[target].rolling(lookback, min_periods = 60).std()
        gradient_scores = (gradient - gradient.rolling(lookback, min_periods = 60).mean()) / gradient.rolling(lookback, min_periods = 60).std()
        
        raw_scores = np.round((regular_scores + gradient_scores).dropna(),3)
        
        scores = np.round((regular_scores + gradient_scores).dropna(),0).astype('int32')
        score_counts = scores.value_counts()

        significant = score_counts.loc[ score_counts >= 10 ]
        minimum = min(significant.index)
        maximum = max(significant.index)
        
        scores = scores.clip(lower=minimum, upper=maximum)

        return [scores, raw_scores]
    
        '''
                target_rate = self.df[target]

        gradient = pd.Series([(grad-target_rate[num])/5 for num, grad in enumerate(target_rate[5:])], index = [i for i in self.df.index[5:]])
        
        # For Rolling Window: False, True
        gradient_means = gradient.rolling(lookback, center=False).mean()
        gradient_std = gradient.rolling(lookback, center=False).std()

        target_rate_means = target_rate.rolling(lookback, center=False).mean()
        target_rate_std = target_rate.rolling(lookback, center=False).std()

        standardized_g = pd.Series((gradient[lookback:] - gradient_means[lookback:])/(gradient_std[lookback:]))
        standardized_df = pd.Series((target_rate[lookback:] - target_rate_means[lookback:])/(target_rate_std[lookback:]))
        
        raw_scores = standardized_g + standardized_df
        
        scores = round(raw_scores, 0)

        return scores.dropna()
        '''

    
    def get_monthly_data(self, data):
        
        return data.groupby(pd.PeriodIndex(data.index, freq="M"))[[i for i in data.columns]].mean()
                

    def get_total_returns(self):
        
        if (self.df[self.sectors[0]] < 1).all().all():
            
            sec_rets = -1*self.df[self.sectors].cumsum().diff(-self.forward).dropna().values
            
            if self.benchmark:
                
                sp_rets = (-1*self.df[self.benchmark].cumsum().diff(-self.forward)).dropna().values
                sec_rets = sec_rets - np.repeat(sp_rets, len(self.sectors)).reshape(sec_rets.shape)
            
            return pd.DataFrame(sec_rets, index = self.df.index[:-self.forward], columns = self.sectors).dropna()
            
        else:
            
            sec_rets = ((-1*self.df[self.sectors].diff(-self.forward)) / self.df[self.sectors]).dropna().values
            
            if self.benchmark:
                sp_rets = ((-1*self.df[self.benchmark].diff(-self.forward)) / self.df[self.benchmark]).dropna().values
                sec_rets = sec_rets - np.repeat(sp_rets, len(self.sectors)).reshape(sec_rets.shape)
            
            return pd.DataFrame(sec_rets, index = self.df.index[:-self.forward], columns = self.sectors).dropna()

    
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
    
    def get_trustworthy_df(self):
        
        cols = [(rate, i) for rate in self.compare_against for i in self.total_returns[rate]]

        truthworthy_df = pd.DataFrame(columns = pd.MultiIndex.from_tuples(cols), index = self.sectors)

        for col in truthworthy_df:
            
            truthworthy_df[col] = normaltest(self.total_returns[col[0]][col[1]])[1] < .05
        
        return truthworthy_df
        
    
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