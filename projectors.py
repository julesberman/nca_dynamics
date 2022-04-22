import numpy as np
from tools import *
import scipy


def ones_model(X, y, dim, reg=0):

    # pad_sig = np.concatenate([np.zeros(dim - 1), sig])
    # X = build_hankel(pad_sig, dim)

    theta = np.ones(dim)
    pred = theta @ X

    return pred, theta


def linear_model(X, y, dim, reg=0):

    # pad_sig = np.concatenate([np.zeros(dim - 1), sig])
    # X = build_hankel(pad_sig, dim)

    lam = reg * np.eye(dim)

    # Get the MLE weights for the LG model
    theta = (y @ X.T) @ np.linalg.inv((X @ X.T) + lam)

    y_lin = theta @ X

    return y_lin, theta


def eigen_proj(X, y, dim, reg=0, shift=1):

    X0 = X[:, :-shift]
    Xp = X[:, shift:]

    lam = reg * np.eye(dim)

    # Get the MLE weights for the LG model
    A = (Xp @ X0.T) @ np.linalg.inv((X0 @ X0.T) + lam)

    evals, evecs = scipy.linalg.eig(A, left=True, right=False)

    # get all real parts
    evals = np.real(np.real_if_close(evals))
    evecs = np.real(np.real_if_close(evecs))

    # get largest eigenvector
    largest_evec = evecs[:, np.nanargmax(evals)]

    # orient eigenvectors same way
    largest_evec *= np.sign(largest_evec[-1])

    # project original series onto largest_evec
    proj_series = largest_evec @ X

    return proj_series, largest_evec


# time window shifted
def eigen_time_proj(sig, y, dim, window=None, reg=0, shift=1, stride=1):

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

    return y, np.mean(np.array(eigs), axis=0)


def DMD_project(X, y, dim, reg=0, shift=1):

    # pad_sig = np.concatenate([np.zeros(dim - 1), sig])
    # X = build_hankel(pad_sig, dim)
    X0 = X[:, :-shift]
    Xp = X[:, shift:]

    r = int(reg)
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

    # normal DMD
    largest_evec = u @ largest_evec_hat

    # exact DMD
    lam_inv = 1/np.max(evals)
    largest_evec = lam_inv * (Xp @ v.T @ s_inv @ largest_evec_hat)

    # multiple by sign of last component??
    largest_evec *= np.sign(largest_evec[-1])

    largest_evec = pick_eigvec_sig(largest_evec, X, y)

    proj_series = largest_evec @ X

    return proj_series, largest_evec


# def TLS_DMD_project(sig, y, dim, reg=0, shift=1):

#     pad_sig = np.concatenate([np.zeros(dim - 1), sig])
#     X = build_hankel(pad_sig, dim)
#     X0 = X[:, :-shift]
#     Xp = X[:, shift:]

#     Z = np.concatenate([X0, Xp]).T

#     k = int(reg)
#     r = dim
#     if k == 0:
#         k = 1

#     u, s, V = np.linalg.svd(Z, full_matrices=False)

#     V11, V21, V12, V22 = V[:r, :k], V[r:, :k], V[:r, k:], V[r:, k:]

#     V11_inv = scipy.linalg.pinv(V11)
#     A = V21 @ V11_inv

#     evals, evecs = scipy.linalg.eig(A, left=True, right=False)

#     # get all real parts
#     evals = np.real(np.real_if_close(evals))
#     evecs = np.real(np.real_if_close(evecs))

#     # get largest eigenvector
#     largest_evec_hat = evecs[:, np.nanargmax(evals)]

#     # normal DMD
#     largest_evec = V[dim:, dim:] @ largest_evec_hat

#     # multiple by sign of last component??
#     largest_evec *= np.sign(largest_evec[-1])

#     largest_evec = pick_eigvec_sig(largest_evec, X, y)

#     proj_series = largest_evec @ X

#     if normalize_output:
#         proj_series = normalize(proj_series)

#     return proj_series, largest_evec


def pick_eigvec_sig(e, X, y):

    pos_err = mean_square_error(e @ X, y)
    neg_err = mean_square_error(-e @ X, y)
    if pos_err < neg_err:
        return e
    else:
        return -e


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


def sia_eig_project_transpose(X_hankel, y, dim, reg=0, shift=1,  normalize_output=True):

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
        X0Xp, X0X0, overwrite_a=True, overwrite_b=True, left=True, right=False)

    # get all real parts
    evals = np.real(np.real_if_close(evals))
    evecs = np.real(np.real_if_close(evecs))

    # get largest eigenvector
    largest_evec = evecs[:, np.nanargmax(evals)]

    # multiple by sign of last component??

    largest_evec = np.flip(largest_evec)
    largest_evec = pick_eigvec_sig(largest_evec, X, y)

    # project original series onto largest_evec
    proj_series = largest_evec @ X

    return proj_series, largest_evec


def eigen_proj_transpose(X, y, dim, reg=0, shift=1):

    X0 = X[:, :-shift]
    Xp = X[:, shift:]

    lam = reg * np.eye(dim)

    # Get the MLE weights for the LG model
    A = (Xp @ X0.T) @ np.linalg.inv((X0 @ X0.T) + lam)

    evals, evecs = scipy.linalg.eig(A, left=False, right=True)

    # get all real parts
    evals = np.real(np.real_if_close(evals))
    evecs = np.real(np.real_if_close(evecs))

    # get largest eigenvector
    largest_evec = evecs[:, np.nanargmax(evals)]

    # orient eigenvectors same way
    largest_evec = np.flip(largest_evec)
    largest_evec = pick_eigvec_sig(largest_evec, X, y)

    # project original series onto largest_evec
    proj_series = largest_evec @ X

    return proj_series, largest_evec
