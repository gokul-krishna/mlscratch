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


def get_metric(name=None):
    if name is not None:
        return getattr(__main__, name)


# Alias
l2_norm = euclidean
l1_norm = city_block
