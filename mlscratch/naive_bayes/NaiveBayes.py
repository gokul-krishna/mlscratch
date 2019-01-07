from ..core import BaseClassifier
from ..imports import *


class NaiveBayesClassifier(BaseClassifier):

    def __init__(self):
        super().__init__()
        self.class_prob = None
        self.cond_prob = None

    def fit(self, X, y):
        self.class_prob = {cls: cls_freq / y.shape[0]
                           for cls, cls_freq in enumerate(np.bincount(y))}

        self.cond_prob = {k: np.sum(X[y == k, :], axis=0) / X[y == k, :].sum()
                          for k in self.class_prob.keys()}

        self.is_fitted = True

    def _predict_one(self, xn):
        xn_prob = [self.class_prob[cls] * np.prod([math.pow(p, t)
                   for t, p in zip(xn, self.cond_prob[cls])])
                   for cls in self.class_prob.keys()]

        return list(self.class_prob.keys())[np.argmax(xn_prob)]

    def predict(self, X_new):
        if self.is_fitted:
            result = [self._predict_one(xn) for xn in X_new]
            return np.array(result)
        else:
            print('Call .fit() method first')
