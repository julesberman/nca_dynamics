import numpy as np
from tools import *
import scipy


class LinearModel:
    def __init__(self, name=''):
        self.name = name

    def train(self, X, y, dim, beta=0):

        # add ones for bias term
        Xb = np.vstack((X, np.ones(X.shape[1])))

        # matrix for regularization
        lam = beta * np.eye(dim + 1)

        # solve the least square problem
        theta = (y @ Xb.T) @ np.linalg.inv((Xb @ Xb.T) + lam)

        # split the filter from the bias
        theta, b = theta[:-1], theta[-1:]

        y_lin = theta @ X + b

        return y_lin, theta, b

    def test(self, X, theta, params):
        b = params
        return theta @ X + b


class ConstantModel:
    def __init__(self, name=''):
        self.name = name

    def train(self, X, y, dim, beta=0):

        theta = np.ones(dim)
        y_hat = theta @ X

        # add one for bias term
        y_hat = np.vstack((y_hat, np.ones(len(y_hat))))

        ab = (y @ y_hat.T) @ np.linalg.inv((y_hat @ y_hat.T))

        a, b = ab[0], ab[1]

        # pass a into theta so filter is scaled correctly
        theta = theta * a

        # final prediction
        pred = theta @ X + b

        return pred, theta, (a, b)

    def test(self, X, theta, params):
        a, b = params
        return theta @ X + b


class EigenModel:
    def __init__(self, name=''):
        self.name = name

    def train(self, X, y, dim, beta=0, shift=1):

        X0 = X[:, :-shift]
        Xp = X[:, shift:]

        # add ones for bias term
        X0b = np.vstack((X0, np.ones(X0.shape[1])))

        # matrix for regularization
        lam = beta * np.eye(dim + 1)

        # solve the least square problem
        A = (Xp @ X0b.T) @ np.linalg.inv((X0b @ X0b.T) + lam)

        # split the filter from the c bias
        A, c = A[:, :-1], A[:, -1:]

        evals, evecs = scipy.linalg.eig(A, left=True, right=False)

        # get all real parts
        evals = np.real(np.real_if_close(evals))
        evecs = np.real(np.real_if_close(evecs))

        # get largest eigenvector
        theta = evecs[:, np.nanargmax(evals)]

        # project original series onto largest_evec
        y_hat = theta @ X

        # solve for scale and bias
        y_hat = np.vstack((y_hat, np.ones(len(y_hat))))
        ab = (y @ y_hat.T) @ np.linalg.inv((y_hat @ y_hat.T))
        a, b = ab[0], ab[1]

        # pass a into theta so filter is scaled correctly
        theta = theta * a

        # final prediction
        pred = theta @ X + b

        return pred, theta, (a, b)

    def test(self, X, theta, params):
        a, b = params
        return theta @ X + b


def sia_eig_project(X_hankel, y, dim, reg=0, shift=1,  normalize_output=True):

    X = X_hankel
    X0 = X[:, :-shift]
    Xp = X[:, shift:]

    X0Xp = X0 @ Xp.T  # / X0.shape[1]
    X0X0 = X0 @ X0.T  # / X0.shape[1]

    # regularize
    lam = reg * np.eye(dim)
    X0X0 += lam

    # solve eigenvalue problem
    evals, evecs = scipy.linalg.eig(
        X0Xp, X0X0, overwrite_a=True, overwrite_b=True)

    # get all real parts
    evals = np.real(np.real_if_close(evals))
    evecs = np.real(np.real_if_close(evecs))

    # get largest eigenvector
    largest_evec = evecs[:, np.nanargmax(evals)]

    # multiple by sign of last component??
    largest_evec *= np.sign(largest_evec[-1])

    # project original series onto largest_evec
    proj_series = largest_evec @ X

    return proj_series, largest_evec
