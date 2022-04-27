import scipy
import numpy as np
from tools import *


def preprocess(signal, factor=1, method='mean'):
    # take mean of all trials
    mean = np.mean(signal, axis=0)
    # down samples mean of trials
    n = len(mean) // factor
    if method == 'nth':
        down = mean[::factor]
    if method == 'fourier':
        down = scipy.signal.resample(mean, n)
    if method == 'mean':
        down = []
        for i in range(0, len(mean), factor):
            m = np.mean(mean[i:i+factor])
            down.append(m)
        down = np.array(down)
    return down


def normalize_01(sig):
    norm = (sig-sig.min())/(sig.max()-sig.min())
    return norm


def normalize_mean_var(sig):
    norm = (sig - sig.mean()) / sig.std()
    return norm


def build_train_test_hankels(X, Y, dim, test_start, train_test_ratio):

    start = int(len(X) * test_start)
    end = start + int(len(X) * train_test_ratio)

    X_test, Y_test = X[start:end],  Y[start:end]
    X_train, Y_train = X.copy(), Y.copy()

    # use to remove from hankel later
    X_train[start:end] = np.nan

    # remove the beginning for causal convoltuion
    Y_train = np.delete(Y_train, np.arange(0, dim))
    Y_test = np.delete(Y_test, np.arange(0, dim))

    test_hankel = build_hankel(X_test, dim)
    train_hankel = build_hankel(X_train, dim)

    # remove lag vector with nan
    nan_cols = np.bitwise_or.reduce(np.isnan(train_hankel), 0)

    train_hankel = np.delete(train_hankel, nan_cols, axis=1)
    Y_train = np.delete(Y_train, nan_cols)

    return train_hankel, Y_train, test_hankel, Y_test


def train_test_method(X, Y, model, dim, train_test_ratio=0.2, betas=np.arange(0, 10, 1)):

    test_start_range = np.linspace(0.0, 1.0-train_test_ratio, 100)
    T = len(test_start_range)
    B = len(betas)

    # X = normalize_01(X)
    # Y = normalize_01(Y)

    train_errors = np.zeros((B, T))
    test_errors = np.zeros((B, T))
    filters = np.zeros((B, T, dim))
    all_params = np.zeros((B, T), dtype='O')
    for t, test_start in enumerate(test_start_range):

        train_hankel, Y_train, test_hankel, Y_test = build_train_test_hankels(
            X, Y, dim, test_start, train_test_ratio)
        for b, beta in enumerate(betas):

            P_train, theta, params = model.train(
                train_hankel, Y_train, dim, beta=beta)
            P_test = model.test(test_hankel, theta, params)

            # if 'Constant' in model.name and beta == 1.0:
            #     plt.plot(Y_test)
            #     plt.plot(P_test)
            #     plt.show()

            train_err = mean_square_error(Y_train, P_train)
            test_err = mean_square_error(Y_test, P_test)

            train_errors[b, t] = train_err
            test_errors[b, t] = test_err
            filters[b, t, :] = theta
            all_params[b, t] = params

    return train_errors, test_errors, filters, all_params, betas
