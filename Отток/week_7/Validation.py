import numpy as np
from sklearn import model_selection
from sklearn import metrics


class CrossValidation:
    
    def __init__(self):
        self.cv = model_selection.StratifiedKFold(n_splits = 4, shuffle = True, random_state = 1)
        
    def cross_validation(self, model, data, y, scoring = 'roc_auc'):
        result = model_selection.cross_val_score(model, data, y, 
                                             cv = self.cv, scoring = scoring,  
                                             n_jobs = -1)
        return result
    
    def cross_validation_for_catboost(self, model, feature_transformer, data, y, metrics, preprocessing = True):
        result = []
        for train_indices, test_indices in self.cv.split(data, y):
            current_model = model
            transformer = feature_transformer
            # train data and target
            if preprocessing != False:
                if type(data).__name__ == 'ndarray':
                    data_train = transformer.fit_transform(data[train_indices])
                    y_train = y[train_indices]
                else:
                    data_train = transformer.fit_transform(data.iloc[train_indices])
                    if type(y).__name__ == 'ndarray':
                        y_train = y[train_indices]
                    else:
                        y_train = y.iloc[train_indices]
                # tets data and target
                if type(data).__name__ == 'ndarray':
                    data_test = transformer.transform(data[test_indices])
                    y_test = y[test_indices]
                else:
                    data_test = transformer.transform(data.iloc[test_indices])
                    if type(y).__name__ == 'ndarray':
                        y_test = y[test_indices]
                    else:
                        y_test = y.iloc[test_indices]
            else:
                # train
                data_train = data[train_indices]
                y_train = y[train_indices]
                # test
                data_test = data[test_indices]
                y_test = y[test_indices]
                
            # fit on train data and predict on dat test
            current_model.fit(data_train, y_train)
            if metrics.__name__ == 'roc_auc_score':
                result.append(metrics(y_test, current_model.predict_proba(data_test)[:, 1]))
            else:
                result.append(metrics(y_test, current_model.predict(data_test)))
        return result