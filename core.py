from .imports import *
from .distance import euclidean


@jit
def _score_classifier(y_pred, y_true):
    """Accuracy"""
    return np.sum(y_true == y_pred, axis=0) / float(y_true.shape[0])


@jit
def _score_regressor(y_pred, y_true):
    """Mean Squared Error"""
    return euclidean(y_pred, y_true)


class BaseClassifier():

    def __init__(self):
        self.is_fitted = False

    def score(self, X, y):
        y_pred = self.predict(X)
        return _score_classifier(y_pred=y_pred, y_true=y)

    def fit(self, X, y):
        pass

    def predict(self, X):
        if self.is_fitted:
            pass

    def __repr__(self):
        pass


class BaseRegressor():

    def __init__(self):
        self.is_fitted = False

    def score(self, X, y):
        y_pred = self.predict(X)
        return _score_regressor(y_pred=y_pred, y_true=y)

    def fit(self, X, y):
        pass

    def predict(self, X):
        if self.is_fitted:
            pass

    def __repr__(self):
        pass
