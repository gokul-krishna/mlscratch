import math
from decimal import *
from ..core import BaseClassifier


def vec_sigmoid(z):
    # python precision safe sigmoid function
    return [(1.0 / (1.0 + float(Decimal(-i).exp()))) for i in z]


def mean(x):
    return sum(x) / len(x)


def vec_logistic_loss(y, y_hat):
    p1 = list(map(lambda x: x[0] * math.log(x[1]), zip(y, y_hat)))
    p2 = list(map(lambda x: (1.0 - x[0]) * math.log(1.0 - x[1]), zip(y, y_hat)))
    p3 = list(map(lambda x: x[0] + x[1], zip(p1, p2)))
    return -mean(p3)


def matmul(X, Y):
    # matrix multiplication implementation
    xw, xh = len(X[0]), len(X)
    yw, yh = len(Y[0]), len(Y)

    if xw == yh:
        res = [[0 for i in range(yw)] for j in range(xh)]
        for i in range(len(X)):
            for j in range(len(Y[0])):
                for k in range(len(Y)):
                    res[i][j] += X[i][k] * Y[k][j]

        return(res)


def list_flatten(x):
    return [item for sublist in x for item in sublist]


def transpose(x):
    # matrix transpose
    return [[x[j][i] for j in range(len(x))] for i in range(len(x[0]))]


def elem_sub(X, C):
    # element wise subtraction for matrix
    return [[item - c for item in sublist] for c, sublist in zip(C, X)]


def get_data(fname):
    # helper function to load data in to an array
    X = []
    y = []
    with open(fname, 'r') as f:
        while True:
            buf = f.readline()
            if not buf:
                break
            d = buf.split(',')
            y.append(int(d[-1]))
            X.append([float(i) for i in d[:-1]])
    return X, y


def _fit_logistic_reg(X, y, learning_rate=0.0001, no_epochs=3000):

    m = len(y)
    # initialize weights and bias
    W = [[0.00001] for i in range(len(X[0]))]
    b = 0.1

    for epoch in range(no_epochs):

        # multiplying with weights
        Z = matmul(X, W)
        # adding bias terms
        Z = [[i[0] + b] for i in Z]
        # get log odds
        y_hat = list(map(lambda x: vec_sigmoid(x), Z))

        # get logistic loss
        loss = vec_logistic_loss(y, list_flatten(y_hat))

        # calculate difference between y and log odds
        dz = list(map(lambda x: [x[1] - x[0]], zip(y, list_flatten(y_hat))))
        t = matmul(transpose(X), dz)

        # calculating gradients for weights and bias
        dw = list(map(lambda x: [(1.0 / m) * x[0]], t))
        db = sum(list_flatten(dz))

        # do gradient descent and update weights & bias
        W = elem_sub(W, list(map(lambda x: x[0] * learning_rate, dw)))
        b = b - learning_rate * db

        if epoch % 100 == 0:
            print(f"loss after {epoch} epoch is: {loss}")

    return W, b


def _predict(X, W, b, thres=0.5):

    # given set of observations, slope and intercept,
    # return hard and soft predictions

    Z = matmul(X, W)
    Z = [[i[0] + b] for i in Z]
    y_hat = list(map(lambda x: vec_sigmoid(x), Z))

    return list(map(lambda x: 1 if x > thres else 0,
                    list_flatten(y_hat))), y_hat


class LogistiRegressionClassifier(BaseClassifier):

    def __init__(self):
        super().__init__()
        self.W = None
        self.b = None

    def fit(self, X, y):
        self.W, self.b = _fit_logistic_reg(X, y)

    def predict(self, X_new):
        return _predict(X_new, self.W, self.b, thres=0.5)
