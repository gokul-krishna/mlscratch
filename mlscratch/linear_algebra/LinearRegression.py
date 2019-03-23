from ..imports import *


def least_squares(x, y):

    x = np.reshape(x, (x.shape[0], 1))
    # add column of 1's
    A = np.concatenate((x, np.ones(x.shape)), axis=1)

    # solving A.T*A*x = A.T*b
    result = np.linalg.solve(np.matmul(A.T, A), np.matmul(A.T, y))
    error = (np.linalg.norm((y - np.matmul(A, result)), ord=2))**2
    slope, intercept = result
    y_model = np.matmul(A, result)

    return slope, intercept, error, x, y, y_model
