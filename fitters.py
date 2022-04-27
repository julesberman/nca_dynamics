import numpy as np
from tools import *
import scipy


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


class EigenModel:
    def __init__(self, name=''):
        self.name = name

    def train(self, X, y, dim, beta=0, shift=1, c_shift=False):

        X0 = X[:, :-shift]
        Xp = X[:, shift:]

        # add ones for bias term
        reg_dim = dim
        if c_shift:
            X0 = np.vstack((X0, np.ones(X0.shape[1])))
            reg_dim += 1

        # matrix for regularization
        lam = beta * np.eye(reg_dim)

        # solve the least square problem
        A = (Xp @ X0.T) @ np.linalg.inv((X0 @ X0.T) + lam)

        if c_shift:
            A, c = A[:, :-1], A[:, -1:]
        else:
            c = 0

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

        return pred, theta, (a, b, c, A)

    def test(self, X, theta, params):
        a, b, c, A = params
        return theta @ X + b


class SingleCEigenModel:
    def __init__(self, name=''):
        self.name = name

    def train(self, X, y, dim, beta=0, shift=1):

        X0 = X[:, :-shift].copy()
        Xp = X[:, shift:].copy()

        # add ones for bias term
        reg_dim = dim

        # matrix for regularization
        lam = beta * np.eye(reg_dim)

        M = X0.T @ np.linalg.inv((X0@X0.T + lam)) @ X0
        S = np.ones_like(M)
        c = (np.mean(Xp) - np.mean(Xp @ M)) / (1-np.mean(S@M))

        # solve the least square problem
        A = ((Xp-c) @ X0.T) @ np.linalg.inv((X0 @ X0.T) + lam)

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

        return pred, theta, (a, b, c, A)

    def test(self, X, theta, params):
        a, b, A = params
        return theta @ X + b


class DMDModel:
    def __init__(self, name='', exact=True):
        self.name = name
        self.exact = exact

    def train(self, X, y, dim, beta=0, shift=1):

        X0 = X[:, :-shift]
        Xp = X[:, shift:]

        r = int(beta)
        if r == 0:
            r = 1

        u, s, v = np.linalg.svd(X0, full_matrices=False)
        u, s, v = u[:, :r], s[:r], v[:r, :]

        s_inv = np.diag(1/s)

        A = u.conj().T @ Xp @ v.conj().T @ s_inv

        evals, evecs = scipy.linalg.eig(A, left=True, right=False)

        # get all real parts
        evals = np.real(np.real_if_close(evals))
        evecs = np.real(np.real_if_close(evecs))

        # get largest eigenvector
        largest_evec_hat = evecs[:, np.nanargmax(evals)]

        if self.exact:
            # exact DMD
            lam_inv = 1/np.max(evals)
            theta = lam_inv * (Xp @ v.T @ s_inv @ largest_evec_hat)
        else:
            # normal DMD
            theta = u @ largest_evec_hat

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
        a, b, = params
        return theta @ X + b
