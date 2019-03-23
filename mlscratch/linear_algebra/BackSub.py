from ..imports import *


def back_sub(A, b):

    if A.shape[0] != A.shape[1]:
        raise Exception('Not a square matrix')

    if A.shape[0] != b.shape[0]:
        raise Exception('Dimentions of matrix b does not match with matrix A')

    if not np.alltrue(np.diagonal(A) != 0):
        raise Exception('Diagonal element is zero')

    if not np.alltrue(A == np.triu(A)):
        raise Exception('Not upper triangular')

    x = np.zeros(A.shape[1])

    for i in reversed(range(A.shape[0])):
        x[i] = (b[i] - np.sum(x[i + 1:] * np.ravel(A[i, (i + 1):]))) / A[i, i]

    return x
