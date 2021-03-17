import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing


class Correlation:
    
    def __init__(self, data):
        self.__data = data
        
    def numeric_correlation(self, num_features, low_limit = 200):
        corr = []
        # scaling numerical features
        scaler = preprocessing.StandardScaler()
        data_num = pd.DataFrame(index = self.__data.index, columns = num_features, 
                                data = scaler.fit_transform(self.__data[num_features]))
        data_num['labels'] = self.__data.labels
        for var in num_features:
            if self.count_element_first_class(var, low_limit):
                corr.append(data_num[data_num['labels'] == 1][var].mean() - 
                            data_num[data_num['labels'] != 1][var].mean())
            else:
                corr.append(np.nan)
        data_num_correlation = pd.DataFrame(index = num_features, 
                                            columns = ['correlation'],
                                            data = corr)
        return data_num_correlation
    
    def categorical_correlation(self, cat_features):
        corr = []
        for var in cat_features:
            if self.__data[var].value_counts().shape[0] < 2:
                corr.append(np.nan)
                continue
            table = pd.crosstab(self.__data[var], self.__data['labels'])
            stat, _, _, _ = stats.chi2_contingency(table)
            corr.append(np.sqrt(stat / table.sum().sum()))
        data_cat_correlation = pd.DataFrame(index = cat_features, 
                                columns = ['correlation'],
                                data = corr)
        return data_cat_correlation
    
    def count_element_first_class(self, var, low_limit):
        return self.__data[self.__data['labels'] == 1][var].dropna().shape[0] > low_limit


