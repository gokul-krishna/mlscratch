from ..core import BaseClassifier
from ..distance import *


@jit
def _predict(X, y, xn, k, dist_fun):
    d = [dist_fun(x, xn) for x in X]
    return d


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

    def predict(self, X_new):
        result = [_predict_one(xn) for xn in X_new]
        return np.array(result)

    def _predict_one(self, xn):
        return _predict(self.X, self.y, xn, self.k, self.dist_fun)
