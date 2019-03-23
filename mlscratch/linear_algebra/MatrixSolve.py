from ..imports import *
from .BackSub import back_sub


def row_swap(X, i, j):

    temp = copy.deepcopy(X[i, :])
    X[i, :] = X[j, :]
    X[j, :] = temp

    return X


def ref(X, basic=True):

    m, n = X.shape

    if basic and (m != (n - 1)):
        raise Exception("Not a square matrix")

    for i in range(m):

        if X[i, i] == 0:
            rows_swapped = False
            for k in range(i, m, 1):
                # finding non zero pivot
                if (X[k, i] != 0) and (not rows_swapped):
                    X = row_swap(X, i, k)
                    rows_swapped = True

            if not rows_swapped:
                raise Exception(f"Can't find a pivot for the row: {i + 1}")

        for j in range(i + 1, m, 1):

            c = (X[j, i] / X[i, i])
            X[j, :] = X[j, :] - (c * X[i, :])

        # print(X)

    return X


def solve(A, b):

    m = A.shape[0]

    Aug_A = np.append(A, b, axis=1)
    Ref_A = ref(Aug_A)
    X = back_sub(Ref_A[:, :m], Ref_A[:, m])

    return X
