from ..imports import *


def SVD_regression(x, y):

    A = np.vstack((x, y)).T
    # transformation to get mean to 0,0
    A_shifted = A - np.array([np.mean(x), np.mean(y)])

    u, s, vh = np.linalg.svd(A_shifted, full_matrices=True)

    # padding with zeros to get mxn dimensions
    s_padded = np.zeros(A.shape)
    s_padded[:2, :2] = s

    # getting rank 1 approximation
    q = u[0, :] @ s_padded[:, 0] * vh[0, :]
    m = q[1] / q[0]

    slope = m
    intercept = -(m * np.mean(x)) + np.mean(y)

    y_model = (m * x) - (m * np.mean(x)) + np.mean(y)

    error = y - y_model

    return slope, intercept, y_model
