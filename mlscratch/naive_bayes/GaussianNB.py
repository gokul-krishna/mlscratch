from ..core import BaseClassifier
from ..imports import *


class GaussianNBClassifier(BaseClassifier):

    def __init__(self):
        super().__init__()
        self.class_prob = None
        self.dict_mean = None
        self.dict_std = None

    def fit(self, X, y):
        self.class_prob = {cls: cls_freq / y.shape[0]
                           for cls, cls_freq in enumerate(np.bincount(y))}

        self.dict_mean = {k: np.mean(X[y == k, :], axis=0)
                          for k in self.class_prob.keys()}

        self.dict_std = {k: np.std(X[y == k, :], axis=0)
                         for k in self.class_prob.keys()}

        self.is_fitted = True

    def _gaussian_pdf(self, x_mean, x_std, x):
        return ((1 / (math.sqrt(2 * math.pi) * x_std)) *
                math.exp(-0.5 * ((x - x_mean) / x_std)**2))

    def _predict_one(self, xn):
        xn_prob = [self.class_prob[cls] * np.prod([self._gaussian_pdf(xm, xs, x)
                   for x, xm, xs in zip(xn, self.dict_mean[cls], self.dict_std[cls])])
                   for cls in self.class_prob.keys()]

        return list(self.class_prob.keys())[np.argmax(xn_prob)]

    def predict(self, X_new):
        if self.is_fitted:
            result = [self._predict_one(xn) for xn in X_new]
            return np.array(result)
        else:
            print('Call .fit() method first')
