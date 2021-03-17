import pandas as pd
from sklearn import impute, preprocessing
from sklearn import pipeline, compose
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class FeaturesBorder:
    
    def __init__(self, data):
        self.__data = data
        
    def choice_features(self, low, hight, num = True, cat_feature_var = 300):    
        features = self.generate_features_name(low, hight)
        result_features = []
        for var in features:
            size_feature = self.__data[var].unique().shape[0]
            if num:
                if self.detect_constant(size_feature):
                    result_features.append(var)
            else:
                if self.detect_constant(size_feature) and self.variation_feature(size_feature, cat_feature_var):
                    result_features.append(var)
        return result_features
    
    def choice_features_with_nan(self, current_features, treshold):
        features = []
        for var in current_features: 
            nan_flag = self.__data[var].isnull().any()
            if nan_flag:
                if self.__proportion_count(var, treshold):
                    features.append(var)
            else:
                features.append(var)
        return features
    
    def __proportion_count(self, var, treshold):
        return sum(self.__data[var].isnull()) / self.__data[var].shape[0] < treshold
    
    @staticmethod
    def generate_features_name(low, hight):
        return [''.join(('Var', str(i))) for i in range(low, hight)]
    
    @staticmethod
    def detect_constant(size_feature):
        return size_feature >= 2
    
    @staticmethod
    def variation_feature(size_feature, cat_feature_var):
        return size_feature < cat_feature_var  
    
    
class SimplePreprocessing(BaseEstimator, TransformerMixin):
    
    def __init__(self, num_features, cat_features):
        self.num_features = num_features
        self.cat_features = cat_features
        
        self.data = None
        self.fit_flag = False
    
        # pipeline for numeric features
        self.num_preprocessing = pipeline.Pipeline(steps = [
            ('num', impute.SimpleImputer(strategy = 'mean')), # strategy = 'constant', fill_value = 0
            ('num_scaler', preprocessing.StandardScaler())
        ])
        # pipeline for numeric features
        self.cat_preprocessing = pipeline.Pipeline(steps = [
            ('cat', impute.SimpleImputer(strategy = 'constant')), # 'most_frequent'
            ('cat_encoder', preprocessing.OneHotEncoder(handle_unknown = 'ignore', sparse = False))
        ])
    
        # transformer for impute NaN and preprocessing features
        self.data_preprocessing = compose.ColumnTransformer(transformers = [
            ('num_features', self.num_preprocessing, self.num_features),
            ('cat_features', self.cat_preprocessing, self.cat_features)
        ])
        
    def fit(self, X, y = None):
        self.data_preprocessing = self.data_preprocessing.fit(X)
        self.fit_flag = True
        return self
        
    def transform(self, X, y = None):
        if self.fit_flag == False:
            raise NotFittedError('This SimplePreprocessing instance is not fitted yet.')
        self.data = self.data_preprocessing.transform(X)
        return self.data
    
    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)
    
    
# Class for get names features for catbooost
class DataForCatboost(BaseEstimator, TransformerMixin):
    
    def __init__(self, num_features, cat_features):
        self.num_features = num_features
        self.cat_features = cat_features
        self.data = None
        self.size_data = 0
        
    def fit(self, X, y = None):
        self.size_data = X.shape[0]
        self.data = pd.DataFrame(columns = self.num_features + self.cat_features, 
                                data = X)
        return self
    
    def transform(self, X, y = None):
        if self.size_data == X.shape[0] or self.size_data == 0:
            return self.data
        else: 
            current_data = pd.DataFrame(columns = self.num_features + self.cat_features, 
                                data = X)
            return current_data
    
    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)
    
    
# Preprocessing for catboost
class PreprocessingForCatboost(BaseEstimator, TransformerMixin):
    
    def __init__(self, num_features, cat_features):
        self.num_features = num_features
        self.cat_features = cat_features
        
        self.data = None
        self.fit_flag = False
        
        self.num_preprocessing = pipeline.Pipeline(steps = [
            ('num', impute.SimpleImputer(strategy = 'mean'))
        ])
        self.cat_preprocessing_for_catboost = pipeline.Pipeline(steps = [
            ('cat_impute', impute.SimpleImputer(strategy = "constant"))
        ])
        
        # трансформер для заполнения пропусков и преобразования вещественных признаков
        self.features_for_catboost = compose.ColumnTransformer(transformers = [
            ('num_features', self.num_preprocessing, self.num_features), 
            ('cat_features', self.cat_preprocessing_for_catboost, self.cat_features) 
        ])
        
        # итоговый pipeline для предобработки данных
        self.all_features = pipeline.Pipeline(steps = [
            ('feature', self.features_for_catboost), 
            ('data', DataForCatboost(self.num_features, self.cat_features))
        ])
        
    def fit(self, X, y = None):
        self.all_features = self.all_features.fit(X)
        self.fit_flag = True
        return self
        
    def transform(self, X, y = None):
        if self.fit_flag == False:
            raise NotFittedError('This SimplePreprocessing instance is not fitted yet.')
        self.data = self.all_features.transform(X)
        return self.data
    
    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)