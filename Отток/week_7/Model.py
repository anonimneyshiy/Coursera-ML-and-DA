import catboost
from sklearn.exceptions import NotFittedError


class Classifier:
    
    def __init__(self, cat_features, data_test_stop, y_test_stop):
        self.flag = False
        self.cat_features = cat_features
        self.data_test_stop = data_test_stop
        self.y_test_stop = y_test_stop
        self.catboost_classifier = catboost.CatBoostClassifier(n_estimators = 850,
                                                               class_weights = [0.9, 0.1], eval_metric = 'AUC', 
                                                               cat_features = self.cat_features)
    def fit(self, X, y):
        self.flag = True
        self.catboost_classifier.fit(X, y,
                        eval_set = (self.data_test_stop, self.y_test_stop),
                        early_stopping_rounds = 200,
                        verbose = False, plot = False);
        return self
    
    def predict(self, X):
        if self.flag is not True:
            raise NotFittedError('This Classifier instance is not fitted yet.')
        return self.catboost_classifier.predict(X)

    def predict_proba(self, X):
        if self.flag is not True:
            raise NotFittedError('This Classifier instance is not fitted yet.')
        return self.catboost_classifier.predict_proba(X)