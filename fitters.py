import torch
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
        lam[-1, -1] = 0

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

    def train(self, X, y, dim, beta=0, shift=1):

        X0 = X[:, :-shift]
        Xp = X[:, shift:]

        # matrix for regularization
        lam = beta * np.eye(dim)
        # lam[-1, -1] = 0

        X0Xp = X0 @ Xp.T
        X0X0 = X0 @ X0.T
        # solve eigenvalue problem
        evals, evecs = scipy.linalg.eig(
            X0Xp, (X0X0+lam), overwrite_a=True, overwrite_b=True)

        # get all real parts
        evals = np.real(np.real_if_close(evals))
        evecs = np.real(np.real_if_close(evecs))

        sortorder = np.argsort(evals)
        # get largest eigenvector
        thetas = evecs[:, sortorder]
        theta = thetas[:, -1]

        # project original series onto largest_evec
        y_hat = theta @ X

        # solve for scale and bias
        y_hat_1 = np.vstack((y_hat, np.ones(len(y_hat))))
        ab = (y @ y_hat_1.T) @ np.linalg.inv((y_hat_1 @ y_hat_1.T))
        a, b = ab[0], ab[1]

        # # pass a into theta so filter is scaled correctly
        theta = theta * a

        # final prediction
        pred = y_hat*a + b

        return pred, theta, (a, b)

    def test(self, X, theta, params):
        a, b = params
        return theta @ X + b


class EigenCShiftModel:
    def __init__(self, name=''):
        self.name = name

    def train(self, X, y, dim, beta=0, shift=1):

        X0 = X[:, :-shift]
        Xp = X[:, shift:]

        Xp = Xp[-1]
        X0 = np.vstack((X0, np.ones(X0.shape[1])))
        reg_dim = dim + 1
        lam = beta * np.eye(reg_dim)
        a = (Xp @ X0.T) @ np.linalg.inv((X0 @ X0.T) + lam)
        a, c = a[:-1], a[-1:]
        A = np.eye(dim, k=1)
        A[-1] = a

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


class EigenTimeModel:
    def __init__(self, name='', window=None):
        self.name = name
        self.window = window

    def train(self, X, y, dim, beta=0, shift=1, stride=1,):

        X0 = X[:, :-shift].copy()
        Xp = X[:, shift:].copy()
        lam = beta * np.eye(dim)

        N = X.shape[1]
        y_hat = np.zeros(N)

        window = self.window
        if window == None:
            window = dim * 4

        for i in range(0, N-window, stride):

            X0 = X0[:, i:i+window]
            Xp = Xp[:, i:i+window]
            X0Xp = X0 @ Xp.T
            X0X0 = X0 @ X0.T
            # solve eigenvalue problem
            w, vl = scipy.linalg.eig(X0Xp, (X0X0+lam))

            theta = vl[:, np.nanargmax(np.abs(w))]
            w = np.sort(np.abs(w))[::-1]

            # project original series onto largest_evec
            proj_series = theta @ X
            y_hat[i:i+len(proj_series)] = proj_series

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


class EigenCompanionModel:
    def __init__(self, name=''):
        self.name = name

    def train(self, X, y, dim, beta=0):

        X0 = X[:, :-1]
        Xp = X[:, 1:]
        Xp = Xp[-1]
        lam = beta * np.eye(dim)

        a = (Xp @ X0.T) @ np.linalg.inv((X0 @ X0.T) + lam)
        A = np.eye(dim, k=1)
        A[-1] = a
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


class EigenL1Model:
    def __init__(self, name=''):
        self.name = name

    def train(self, X, y, dim, beta=0, shift=1):

        X0 = X[:, :-shift]
        Xp = X[:, shift:]

        X0Xp = X0 @ Xp.T
        X0X0 = X0 @ X0.T

        A_init = (X0Xp @ X0X0.T) @ np.linalg.inv((X0X0 @ X0X0.T))

        SolL1T = torch.tensor(A_init, requires_grad=True)
        X0T = torch.tensor(X0)
        opt = torch.optim.Adam([SolL1T], lr=0.001)
        X0T = torch.tensor(X0)
        XpT = torch.tensor(Xp)
        loss_hist = []
        for _ in range(10000):
            loss = (XpT - SolL1T @ X0T).abs().sum()
            loss_hist.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        A = SolL1T.data.numpy()
        evals, evecs = scipy.linalg.eig(A, left=True, right=False)

        # get all real parts
        evals = np.real(np.real_if_close(evals))
        evecs = np.real(np.real_if_close(evecs))

        sortorder = np.argsort(evals)
        # get largest eigenvector
        thetas = evecs[:, sortorder]
        theta = thetas[:, -1]

        # project original series onto largest_evec
        y_hat = theta @ X

        # solve for scale and bias
        y_hat_1 = np.vstack((y_hat, np.ones(len(y_hat))))
        ab = (y @ y_hat_1.T) @ np.linalg.inv((y_hat_1 @ y_hat_1.T))
        a, b = ab[0], ab[1]

        # # pass a into theta so filter is scaled correctly
        theta = theta * a

        # final prediction
        pred = y_hat*a + b

        return pred, theta, (a, b)

    def test(self, X, theta, params):
        a, b = params
        return theta @ X + b


def L1_solve(X0, Xp):

    X0Xp = X0 @ Xp.T
    X0X0 = X0 @ X0.T

    A = (X0Xp @ X0X0.T) @ np.linalg.inv((X0X0 @ X0X0.T))

    SolL1T = torch.tensor(A, requires_grad=True)
    X0T = torch.tensor(X0)
    opt = torch.optim.Adam([SolL1T], lr=0.001)
    X0T = torch.tensor(X0)
    XpT = torch.tensor(Xp)
    loss_hist = []
    for _ in range(10000):
        loss = (XpT - SolL1T @ X0T).abs().sum()
        loss_hist.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    SolL1 = SolL1T.data.numpy()
    return SolL1


class DMDModel:
    def __init__(self, name='', exact=False):
        self.name = name
        self.exact = exact

    def train(self, X, y, dim, beta=0, shift=1):

        X0 = X[:, :-shift]
        Xp = X[:, shift:]

        r = int(beta)
        # if r % 2 == 1:
        #     r += 1
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

        return pred, theta, (a, b, A)

    def test(self, X, theta, params):
        a, b, A = params
        return theta @ X + b
