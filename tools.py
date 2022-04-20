import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random
import math
from collections.abc import Iterable
import scipy
import scipy.io
from scipy import signal
import seaborn as sns


def set_seaborn():
    sns.set()
    sns.color_palette("tab10")
    FIG_SIZE = (14, 8)

    color_list = sns.color_palette("tab10")

    sns.set_theme(rc={'axes.prop_cycle': plt.cycler(color=color_list),
                      'lines.linewidth': 1.6,
                      'lines.markersize': 10,
                      'font.family': 'Arial',
                      'font.size': 16,
                      'axes.titlecolor': 'black',
                      'axes.titleweight': 'bold',
                      'axes.labelcolor': 'black',
                      'legend.facecolor': 'white',
                      'legend.edgecolor': 'black',
                      'figure.dpi': 100,
                      'figure.figsize': FIG_SIZE
                      })


def plot_3d(X, title="", size=(5, 12)):
    fig = plt.figure()
    fig.set_size_inches(*size)
    ax = plt.axes(projection='3d')
    ax.plot3D(X[0], X[1], X[2], 'blue', '.')
    plt.title(f'{title}')
    plt.show()


def plotly_3d(X, title="", mode='markers'):

    fig = go.Figure(data=go.Scatter3d(
        x=X[0], y=X[1], z=X[2],
        mode=mode,
        marker=dict(
            size=1,
            color=np.arange(0, len(X[0])),
            colorscale='Viridis',
        ),
    ))
    fig.update_layout(
        width=1000,
        height=1000,
        title=title
    )
    fig.show()


def gen_hankel(X, N, tao):
    X = X[::tao]
    size = len(X)-N
    H = np.zeros((N, size))
    lag_vec = np.arange(0, N)
    for i in range(0, size):
        H[:, i] = X[lag_vec+i]

    return H


def random_walk_coords(n):

    x = np.zeros(n).astype(int)
    y = np.zeros(n).astype(int)

    # filling the coordinates with random variables
    for i in range(1, n):
        val = random.randint(1, 4)
        if val == 1:
            x[i] = x[i - 1] + 1
            y[i] = y[i - 1]
        elif val == 2:
            x[i] = x[i - 1] - 1
            y[i] = y[i - 1]
        elif val == 3:
            x[i] = x[i - 1]
            y[i] = y[i - 1] + 1
        else:
            x[i] = x[i - 1]
            y[i] = y[i - 1] - 1

    return x, y


def rand_sgn():
    return 1 if random.random() < 0.5 else -1


def rand_float(low, high):
    return random.random()*(high-low) + low


def gaussian(N, mu=0, sigma=0.1, normalize=True):
    x = np.linspace(-1, 1, N)
    y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2))

    if normalize:
        y /= np.max(y)

    return y


def build_signal(N, f, spike_locations, noise=0, normalize=True, return_spikes=False):
    Y = np.zeros(N)
    spikes = np.zeros(N)

    for s in spike_locations:
        spike = f()
        width = len(spike)
        if s+width <= N:
            Y[s:s+width] = spike
            spikes[s:s+width] = np.linspace(1, 0, width)  # 1

    Y += noise * np.random.normal(size=N, scale=noise)

    if normalize:
        Y -= np.mean(Y)
        Y /= np.max(Y)

    if return_spikes:
        return Y, spikes

    return Y


def build_signal_grid(x, num, noise=0, a=[1, 1], lam=[1, 1], return_basis=False, center=False):
    final_sig = np.zeros_like(x)
    basis = []
    for i in range(num):
        c = random.choice(x)
        sgn = rand_sgn()
        sig = tanh(x, a=rand_float(a[0], a[1]), lam=rand_float(
            lam[0], lam[1]), center=c)[::sgn]
        final_sig += sig
        if return_basis:
            basis.append(sig)
    final_sig += noise * np.random.normal(size=len(x), scale=1)

    if center:
        final_sig -= np.mean(final_sig)

    if return_basis:
        return final_sig, basis

    return final_sig


# FROM https://github.com/sethhirsh/sHAVOK/blob/master/Figure%201.ipynb
def build_hankel(data, rows, cols=None):
    if cols is None:
        cols = len(data) - rows
    X = np.empty((rows, cols))
    for k in range(rows):
        X[k, :] = data[k:cols + k]
    return X


def multi_build_hankel(X_ts, N):
    Hs = []
    M = len(X_ts)
    L = len(X_ts[0])-N
    for data in X_ts:
        H = build_hankel(data, N, L)
        Hs.append(H)

    multi_H = np.zeros((L, N*M))
    for m in range(M):
        H = Hs[m]
        for n in range(N):
            multi_H[:, m*n] = H[n, :]

    return multi_H


def HAVOK(X, dt, r, norm, center=False, return_uv=False):
    if (center):
        m = X.shape[0]
        X̄ = X - X[m//2, :]
        U, Σ, Vh = np.linalg.svd(X̄, full_matrices=False)
    else:
        U, Σ, Vh = np.linalg.svd(X, full_matrices=False)
    V = Vh.T
    # polys = true_polys(X.shape[0], dt, r, center)
    # for _i in range(r):
    #     if (np.dot(U[:, _i], polys[:, _i]) < 0):
    #         U[:, _i] *= -1
    #         V[:, _i] *= -1
    V1 = V[:-1, :r]
    V2 = V[1:, :r]
    A = (V2.T @ np.linalg.pinv(V1.T) - np.eye(r)) / (norm * dt)
    if (return_uv):
        return A, U, V
    return A


def true_polys(rows, dt, r, center):
    m = rows // 2
    Ut = np.linspace(-m*dt, m*dt, rows)
    poly_stack = []
    for j in range(r):
        if (center):
            poly_stack.append(Ut ** (j + 1))
        else:
            poly_stack.append(Ut ** j)
    poly_stack = np.vstack(poly_stack).T
    Q = np.empty((rows, r))  # Perform Gram-Schmidt
    for j in range(r):
        v = poly_stack[:, j]
        for k in range(j - 1):
            r_jk = Q[:, k].T @ poly_stack[:, j]
            v -= (r_jk * Q[:, k])
        r_jj = np.linalg.norm(v)
        Q[:, j] = v / r_jj
    return Q


def sigmoid(N, width):
    '''
    Returns array of a horizontal mirrored normalized sigmoid function
    output between 0 and 1
    Function parameters a = center; b = width
    '''
    b = width
    x = np.linspace(-1, 1, N)
    s = 1/(1+np.exp(b*(x)))
    s = 1*(s-min(s))/(max(s)-min(s))
    ss = np.zeros(N*2)
    ss[:N] = s
    ss[N:] = s[::-1]
    return ss-1


def tanh(t, a=1, lam=1, center=0):
    return a/(1+np.exp(lam*-(t+center)))


def plot_embed(data, Ns, hankel=False, sz=(20, 14), c=None, line=False, center=False, square_plot=False):

    num_sl = len(Ns)
    r = math.ceil(math.sqrt(num_sl))

    fig, axarr = plt.subplots(r, r)
    fig.set_size_inches(sz[0], sz[1])
    if square_plot:
        fig.set_size_inches(sz[1], sz[1])

    if isinstance(axarr, Iterable):
        axs1 = [item for sublist in axarr for item in sublist]
    else:
        axs1 = [axarr]

    for i, N in enumerate(Ns):

        if hankel:
            H = data
        else:
            H = build_hankel(data, N, len(data)-N)

        if center:
            H -= H[H.shape[0]//2]

        u, s, v = np.linalg.svd(H, full_matrices=False)

        H_hat = v[:2]

        # H_hat[0] -= np.mean(H_hat[0])
        # H_hat[1] -= np.mean(H_hat[1])
        # H_hat[0] /= np.max(H_hat[0])
        # H_hat[1] /= np.max(H_hat[1])
        L = len(H_hat[0])

        # plt.plot(s[0:10], '.-',)
        # plt.title("Hankel Singular Values")
        # plt.show()

        # plt.plot(u[:, :3])
        # plt.show()

        # plt.plot(v[0])
        # plt.plot(v[1])
        # plt.plot(trajectory[:,1])
        # plt.plot(trajectory[:,0])
        # plt.show()
        if c is None:
            c = np.arange(0, L)
        c = c[:L]

        scatter = axs1[i].scatter(H_hat[0], H_hat[1], c=c)

        if square_plot:
            mi, mx = np.min(H_hat), np.max(H_hat)
            plt.xlim([mi, mx])
            plt.ylim([mi, mx])

        if line:
            axs1[i].plot(H_hat[0], H_hat[1])
        axs1[i].set_title(f'N={N}')

    if c is not None:
        fig.colorbar(scatter)

    plt.show()


def normalize(x):
    new = x.copy()
    new -= np.mean(new)
    new /= np.max(np.abs(new))
    new -= np.mean(new)
    return new


def normalize_l2(x):
    x -= np.mean(x)
    x /= np.linalg.norm(x)
    return x


def read_lmc(file, pre_process=False, downsample=10):
    data = np.array(scipy.io.loadmat(f'./lmc/{file}')['DATAFILE'])
    ress = data[:, ::2]
    stims = data[:, 1::2]
    ress = np.swapaxes(ress, 0, 1)
    stims = np.swapaxes(stims, 0, 1)

    if pre_process:
        ress = normalize(np.mean(ress[:, ::downsample], axis=0))
        stims = normalize(np.mean(stims[:, ::downsample], axis=0))

    return ress, stims


def cross_corr(x, y):
    corr = signal.correlate(x, y)
    lags = signal.correlation_lags(len(x), len(y))
    return lags, corr


def max_eig_project(sig, N, shift=1, norm=1):

    # build Hankels for X, X_0 and X_+
    X = build_hankel(sig, N)
    X0 = X[:, :-shift]
    Xp = X[:, shift:]

    X0Xp = X0 @ Xp.T  # / X0.shape[1]
    X0X0 = X0 @ X0.T  # / X0.shape[1]

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


# time window shifted
def stride_eig(sig, window, dim, shift=1, stride=1, return_eigs=False):

    N = len(sig)
    y = np.zeros(N)

    eigs = []
    for i in range(0, N-window, stride):

        sigwin = sig[i:i+window]
        proj_series, evec = max_eig_project(sigwin, dim, shift=shift)
        y[i:i+len(proj_series)] = proj_series  # evec @ sigwin[-dim:]
        if return_eigs:
            eigs.append(evec)

    if return_eigs:
        return y, np.array(eigs)
    return y


def low_rank_approx(A, r=1):
    """
    Computes an r-rank approximation of a matrix
    given the component u, s, and v of it's SVD
    Requires: numpy
    """

    u, s, v = np.linalg.svd(A, full_matrices=False)
    Ar = u[:, :r] * s[:r] @ v[:r, :]
    return Ar


def mean_square_error(A, B): return np.square(np.subtract(A, B)).mean()
