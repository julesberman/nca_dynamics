import numpy as np
from tools import *
import scipy
import pydmd


def ones_model(sig, y, dim, reg=0, normalize_output=True):

    pad_sig = np.concatenate([np.zeros(dim - 1), sig])
    X = build_hankel(pad_sig, dim)

    theta = np.ones(dim)
    y_lin = theta @ X

    if normalize_output:
        y_lin = normalize(y_lin)

    return y_lin, theta


def linear_model(sig, y, dim, reg=0, normalize_output=True):

    pad_sig = np.concatenate([np.zeros(dim - 1), sig])
    X = build_hankel(pad_sig, dim).T

    lam = reg**2 * np.eye(dim)

    # Get the MLE weights for the LG model
    theta = np.linalg.inv((X.T @ X) + lam) @ X.T @ y

    y_lin = X @ theta

    if normalize_output:
        y_lin = normalize(y_lin)

    return y_lin, theta


def eigen_proj(sig, y, dim, reg=0, shift=1, normalize_output=True):

    pad_sig = np.concatenate([np.zeros(dim - 1), sig])
    X = build_hankel(pad_sig, dim).T
    X0 = X[:-shift]
    Xp = X[shift:]

    lam = reg**2 * np.eye(dim)

    # Get the MLE weights for the LG model
    A = np.linalg.inv((X0.T @ X0) + lam) @ X0.T @ Xp

    # # regularize
    # X0 = np.concatenate([X0, reg**2*np.eye(dim)])
    # Xp = np.concatenate([Xp, np.zeros((dim, dim))])

    # A, _, _, _ = scipy.linalg.lstsq(X0, Xp)

    evals, evecs = scipy.linalg.eig(A, left=False, right=True)

    # get all real parts
    evals = np.real(np.real_if_close(evals))
    evecs = np.real(np.real_if_close(evecs))

    # get largest eigenvector
    largest_evec = evecs[:, np.nanargmax(evals)]

    # multiple by sign of last component??
    largest_evec *= np.sign(largest_evec[-1])

    # project original series onto largest_evec
    proj_series = X @ largest_evec

    if normalize_output:
        proj_series = normalize(proj_series)

    return proj_series, largest_evec


def eigen_proj_low_rank(sig, y, dim, reg=0, shift=1, normalize_output=True):

    pad_sig = np.concatenate([np.zeros(dim - 1), sig])
    X = build_hankel(pad_sig, dim)
    X0 = X[:, :-shift].T
    Xp = X[:, shift:].T

    A, _, _, _ = scipy.linalg.lstsq(X0, Xp)

    reg = int(reg)
    if reg > 0:
        A = low_rank_approx(A, r=reg)

    evals, evecs = scipy.linalg.eig(A, left=False, right=True)

    # get all real parts
    evals = np.real(np.real_if_close(evals))
    evecs = np.real(np.real_if_close(evecs))

    # get largest eigenvector
    largest_evec = evecs[:, np.nanargmax(evals)]

    # multiple by sign of last component??
    largest_evec *= np.sign(largest_evec[-1])

    # project original series onto largest_evec
    proj_series = largest_evec @ X

    if normalize_output:
        proj_series = normalize(proj_series)

    return proj_series, largest_evec


# time window shifted
def eigen_time_proj(sig, y, dim, window=None, reg=0, shift=1, stride=1, normalize_output=True):

    N = len(sig)
    y = np.zeros(N)

    if window == None:
        window = dim * 4
    eigs = []
    for i in range(0, N-window, stride):

        sigwin = sig[i:i+window]
        proj_series, evec = eigen_proj(sigwin, y, dim, reg=reg, shift=shift)
        y[i:i+len(proj_series)] = proj_series  # evec @ sigwin[-dim:]
        eigs.append(evec)

    if normalize_output:
        y = normalize(y)

    return y, np.mean(np.array(eigs), axis=0)


def DMD_project(sig, y, dim, reg=0, shift=1, normalize_output=True):

    pad_sig = np.concatenate([np.zeros(dim - 1), sig])
    X = build_hankel(pad_sig, dim)
    X0 = X[:, :-shift]
    Xp = X[:, shift:]

    r = reg
    u, s, v = np.linalg.svd(X0, full_matrices=False)
    u, s, v = u[:, :r], s[:r], v[:r, :]

    s_inv = np.diag(1/s)

    A = u.conj().T @ Xp @ v.conj().T @ s_inv

    evals, evecs = scipy.linalg.eig(A, left=True, right=False)

    # get all real parts
    evals = np.real(np.real_if_close(evals))
    evecs = np.real(np.real_if_close(evecs))

    # get largest eigenvector
    largest_evec = evecs[:, np.nanargmax(evals)]

    # multiple by sign of last component??
    largest_evec *= np.sign(largest_evec[-1])

    # project original series onto largest_evec
    proj_series = np.convolve(sig, largest_evec)

    if normalize_output:
        proj_series = normalize(proj_series)

    return proj_series, largest_evec
