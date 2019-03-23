from ..imports import *

delta = 0.85


def convert_to_adj(data):
    blogs = np.unique(data)
    # creating map between blog and blog id
    blogid = {n: i for i, n in enumerate(blogs)}
    inv_blogid = {v: k for k, v in blogid.items()}
    n = blogs.size

    numdata = np.vectorize(blogid.get)(data)

    A = np.zeros((n, n))
    for source, destination in numdata:
        A[destination, source] = 1

    return A, inv_blogid, n


def normalize_fix_dangling(X):

    n = X.shape[0]

    # to fix dangling page problem
    for i in np.argwhere(np.sum(X, axis=0) == 0).ravel():
        X[i, i] = 1

    Colsum = np.sum(X, axis=0)

    return X / np.outer(np.ones(n), Colsum)


# return A^n stop when the error is with in permissible value
def Anx(A, x0, n):
    max_iter = 1000
    x = x0
    counter = 1

    while (max(abs(A @ x - x)) > 10**(-10)) & (counter < max_iter):
        x = A @ x
        counter += 1

    print("Total iterations: ", counter)

    return x


def page_rank(data, top_n=10):

    A, inv_blogid, n = convert_to_adj(data)
    B = normalize_fix_dangling(A)
    # To fix pages with 0 outgoing links
    M = delta * B + ((1 - delta) / n) * np.ones([n, n])
    P = Anx(M, np.ones(n) / n, 100)
    max_idx = np.argsort(P)[-top_n:]
    return np.vectorize(inv_blogid.get)(max_idx)
