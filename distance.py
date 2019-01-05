from .imports import *


@jit
def city_block(p, q):
    return np.sum(np.abs(p - q))


@jit
def euclidean(p, q):
    return np.sqrt(np.sum(np.power((p - q), 2)))


@jit
def minkowski(p, q, power):
    return np.power(np.sum(np.power(np.abs(p - q), power)), 1.0 / power)


@jit
def chebyshev(p, q):
    return np.max(np.abs(p - q))


# Alias
l2norm = euclidean
l1norm = city_block
