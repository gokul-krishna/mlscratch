from ..distance import get_metric, euclidean
from ..imports import *
euclidean

class kMeans():

    def __init__(self, k=2, dist_metric='euclidean'):
        self.dist_metric = dist_metric
        self.dist_fun = get_metric(dist_metric)
        self.k = k
        self.tol = 1e-4
        self.max_iter = 1000

    def _normalize(self, x):
        return ((x - x.mean(axis=0)) / x.std(axis=0))

    def fit_predict(self, X):

        X = self._normalize(X)
        centers = np.random.randn(self.k, X.shape[1])
        tol_achived = False
        iter_no = 0

        while (not tol_achived) and iter_no < self.max_iter:

            distances = np.array([[self.dist_fun(xi, c) for c in centers]
                                  for i, xi in enumerate(X)])
            clusters = distances.argmin(axis=1)
            new_centers = np.array([X[clusters == i].mean(axis=0) for i in range(self.k)])
            center_changes = [euclidean(ec, [0, 0]) for ec in (centers - new_centers)]
            tol_achived = np.alltrue((np.less_equal(center_changes, self.tol)))
            centers = new_centers
            iter_no += 1

        return clusters
