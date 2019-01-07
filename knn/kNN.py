from ..core import BaseClassifier
from ..distance import get_metric
from ..imports import *


def _predict(X, y, xn, k, dist_fun):
    d = np.array([dist_fun(xn, xi) for xi in X])
    return np.argmax(np.bincount(y[np.argsort(d)][:k]))


class kNNClassifier(BaseClassifier):

    def __init__(self, k=1, dist_metric='euclidean'):
        super().__init__()
        self.dist_metric = dist_metric
        self.dist_fun = get_metric(dist_metric)
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.is_fitted = True

    def _predict_one(self, xn):
        return _predict(self.X, self.y, xn, self.k, self.dist_fun)

    def predict(self, X_new):
        if self.is_fitted:
            result = [self._predict_one(xn) for xn in X_new]
            return np.array(result)
        else:
            print('Call .fit() method first')
