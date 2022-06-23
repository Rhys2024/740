from operator import index
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import yfinance as yf
import fredapi as fa
from datetime import date, timedelta
from Modules.Exposure_Report import Exposure
import numpy as np
import matplotlib.pyplot as plt
import Modules.Useful_Functions as u
from statsmodels.tsa.stattools import adfuller
from scipy.stats.mstats import normaltest
import seaborn as sns
import FileFinder.FileFinder as ff

### ML REQUIREMENTS ###
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")

fred = fa.Fred('4fb0ce271d0f66f4b5b3904b4aaf1dd0')


class MacroStrat1(object):
    

    def __init__(self, model_look_back = 252, model_forward = 30, compare_against = None, benchmark = None):
        
        self.compare_against = compare_against
        self.__possible_compares = ["Real Yield", "Yield Curve", "inflation",
                                  "uncertainty_index", "market_volume", "market_volatility", "SP-500"]
        
        cosidered = set(self.compare_against).intersection(set(self.__possible_compares))
        not_considered = set(self.__possible_compares).difference(set(self.compare_against))
        self.benchmark = benchmark
        if self.benchmark in not_considered:
            not_considered.remove(self.benchmark)
        
        assert isinstance(self.compare_against, list), "input 'compare_against' must be a list"
        assert cosidered == set(self.compare_against), f"values in list 'compare_against' must be in the list of possible compares, see 'self.__possible_compares' "

        
        self.model_forward = model_forward
        self.model_lookback = model_look_back
        
        macro_data_path = ff.get_path("macro_data.csv", "Reference_Data")
        self.sector_df = pd.read_csv(macro_data_path, index_col="Date").drop(columns = list(not_considered))
            
        self.sector_data = Exposure(self.sector_df, self.model_lookback, self.model_forward, self.compare_against, self.benchmark, False)
        self.__sectors = self.sector_data.sectors
        self.num_sectors = len(self.__sectors)
        
        self.sector_labels = {sec : num for num, sec in enumerate(self.sector_data.sectors)}
        self.sector_labels_reverse = {num : sec for num, sec in enumerate(self.sector_data.sectors)}
        self.feature_columns = list(self.sector_data.forward_returns_daily.columns[-len(self.compare_against):])
        
    
    def run_model(self, model = "Random Forest", cutoff = .7):
        
        self.sector_return_data = self.sector_data.forward_returns_daily.dropna().copy()
        
        cutoff = int(len(self.sector_return_data)*cutoff)
        
        self.df_for_assesment = self.sector_return_data.copy().iloc[:,:self.num_sectors][cutoff:]
        self.__set_consideration_data()
        
        variables = self.set_variables(cutoff = cutoff)
        fitted_model = self.random_forest(variables)
        model_predictions = self.prediction(variables, fitted_model)
        
        confusion = confusion_matrix(variables['YTEST'], model_predictions)
        
        assessment = self.__assess_model(variables, model_predictions, cutoff)
        
        return {"assessment" : assessment,
                "best_features" : self.important_features(fitted_model),
                "prediction_accuracy_per_sector" : self.inDepth_accuracy_report(confusion),
                "annualized_mean_return" : assessment.predicted_top_sector_returns.mean() * (252/self.model_forward),
                "confusion" : pd.DataFrame(confusion, index=self.__sectors, columns=self.__sectors)}
    
    
    def __set_consideration_data(self):
        
        self.sector_return_data['top_sector'] = [self.sector_return_data.iloc[:,:self.num_sectors].loc[i].sort_values().index[-1] 
                                                 for i in self.sector_return_data.iloc[:,:self.num_sectors].index]
        self.sector_return_data['top_sector'] = [self.sector_labels[i] for i in self.sector_return_data['top_sector']]
        self.sector_return_data['top_sector_returns'] = self.sector_return_data[self.__sectors].max(axis=1)
        self.sector_return_data = self.sector_return_data.iloc[:,-len(self.compare_against)-2:]
        
        
    def set_variables(self, cutoff):
        
        temp = self.feature_columns.copy()
        
        if len(self.feature_columns) == 1:
            temp = temp[0]
        
        return {'X' : self.sector_return_data[temp][:cutoff].values,
                'Y' : self.sector_return_data['top_sector'][:cutoff].values,
                'XTEST' : self.sector_return_data[temp][cutoff:].values,
                'YTEST' : self.sector_return_data['top_sector'][cutoff:].values}
        
    def random_forest(self, variables, estimators = 100):
        return RandomForestClassifier(n_estimators = estimators).fit(variables['X'],variables['Y'])
    
    def prediction(self, variables, fitted_model):
        return fitted_model.predict(variables['XTEST'])
    
    def regularize(self):
        pass
    
    def important_features(self, fitted_model):
        
        return pd.Series(fitted_model.feature_importances_, index = self.feature_columns)
        
    def __assess_model(self, variables, preds, cutoff):
        
        self.df_for_assesment['predicted_top_sector'] = list(map(lambda n : self.sector_labels_reverse[n], preds))
        self.df_for_assesment['actual_top_sector'] = list(map(lambda n : self.sector_labels_reverse[n], variables['YTEST']))
        self.df_for_assesment['predicted_top_sector_returns'] = list(map(lambda sec, n : self.df_for_assesment[self.sector_labels_reverse[sec]].iloc[n], 
                                                                 preds, range(len(preds))))
        self.df_for_assesment['actual_top_sector_returns'] = self.sector_return_data[cutoff:]['top_sector_returns']
        self.df_for_assesment = self.df_for_assesment.iloc[:,self.num_sectors:]
        
        return self.df_for_assesment
   
    def inDepth_accuracy_report(self, confusion):
        
        guess_accs = np.diag(confusion) / np.sum(confusion, axis=0)
        
        return pd.Series(guess_accs, index = self.__sectors)

    def confusion_matrix(self, confusion):
        
        plt.figure(figsize=(25,20))
        fig = sns.heatmap(confusion, annot=True, annot_kws = {'fontsize' : 24}, cmap='Blues')
        fig.set_xticklabels(self.__sectors, size = 14)
        fig.set_yticklabels(self.__sectors, size = 14)
        fig.set_xlabel("Predicted", size = 24)
        fig.set_ylabel("Actual", size = 24)
        plt.tight_layout()
        plt.show()