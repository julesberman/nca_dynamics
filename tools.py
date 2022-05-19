import types
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random
import math
from collections.abc import Iterable
import scipy
import scipy.io
from scipy import signal
import itertools
import pandas as pd
import seaborn as sns


def set_seaborn(params={}):
    sns.set()
    sns.color_palette("mako")
    sns.set_style('white')



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


def build_signal_grid(x, noise=0, a_s=[1], l_s=[1], c_s=[0], s_s=[1], return_basis=False, center=False):
    final_sig = np.zeros_like(x)
    basis = []
    num = len(a_s)
    for i in range(num):
        a, l, c, s = a_s[i], l_s[i], c_s[i], int(s_s[i])
        sig = tanh(x, a=a, lam=l, center=c)[::s]
        sig += (noise * np.random.randn(len(sig)))
        final_sig += sig
        if return_basis:
            basis.append(sig)

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


def factor_int_close_to_square(n):
    d = math.ceil(math.sqrt(n))
    opt = math.inf
    opt_o = (0, 0)
    off = [(0, 0), (0, -1), (-1, 1), (-2, 1), (-2, 0)]
    for (l, r) in off:
        extra = ((d+l)*(d+r)) - n
        if extra < opt and extra >= 0:
            opt = extra
            opt_o = (l, r)
    ans = [d+opt_o[0], d+opt_o[1]]
    ans.sort()
    return tuple(ans)


def make_subplt_arr(d1, d2=None):
    if d2 is None:
        d1, d2 = factor_int_close_to_square(d1)
    fig, axarr = plt.subplots(d1, d2)

    if isinstance(axarr, Iterable):
        if isinstance(axarr[0], Iterable):
            axs1 = [item for sublist in axarr for item in sublist]
        else:
            axs1 = axarr
    else:
        axs1 = [axarr]

    return fig, axs1


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


def lorentz(T=20, h=0.01):
    def init_XYZ(m):
        X = np.zeros(m)
        Y = np.zeros(m)
        Z = np.zeros(m)
        X[0] = -5.91652
        Y[0] = -5.52332
        Z[0] = 24.57231
        return (X, Y, Z)

    m = int(T/h)
    (X, Y, Z) = init_XYZ(m)

    for k in range(0, m-1):
        X[k+1] = X[k] + h*10*(Y[k]-X[k])
        Y[k+1] = Y[k] + h*((X[k]*(28-Z[k]))-Y[k])
        Z[k+1] = Z[k] + h*(X[k]*Y[k]-8*Z[k]/3)

    return np.array([X, Y, Z])


def lorentz_signal(T=20, h=0.01, noise=0.0):
    L = lorentz(T, h)

    obs_x = L[0, :]
    obs_y = L[1, :]
    obs_z = L[2, :]

    X = obs_x
    X = X + (noise * np.random.randn(len(X)))

    return X


def make_histogram(data, bins=20, title='', xlabel='', ylabel='Counts', logscale=False):

    bins = np.linspace(math.ceil(min(data)),
                       math.floor(max(data)),
                       bins)  # fixed number of bins

    plt.xlim([min(data)-5, max(data)+5])

    plt.hist(data, bins=bins, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if logscale:
        plt.yscale('log', nonposy='clip')

    plt.show()


def normalize(timeseries):
    # new = x.copy()
    # new -= np.mean(new)
    # new /= np.max(np.abs(new))
    # return new
    return (timeseries-timeseries.min())/(timeseries.max()-timeseries.min())


def normalize_l2(x):
    x -= np.mean(x)
    x /= np.linalg.norm(x)
    return x


def read_lmc(file):
    data = np.array(scipy.io.loadmat(f'./lmc/{file}')['DATAFILE'])
    ress = data[:, ::2]
    stims = data[:, 1::2]
    ress = np.swapaxes(ress, 0, 1)
    stims = np.swapaxes(stims, 0, 1)

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


def mean_square_error(A, B): return np.square(np.subtract(A, B)).mean()


def exp_f(x, e, a): return a*np.exp(x*e)


def build_exp_series(a_s, e_s, noise=0.0, time=np.arange(0, 3, 0.1)):
    components = []
    X = np.zeros_like(time)
    for i in range(len(e_s)):
        c = exp_f(time, e_s[i], a_s[i])
        c = c.real
        X += c
        components.append(c)

    X *= (1+noise * np.random.randn(len(X)))
    y_i = np.argmax(np.abs(e_s.real))
    Y = components[y_i]

    return X, Y, time, components


def arr_if_scalar(e):
    if isinstance(e, Iterable) and type(e) is not str:
        return e
    else:
        return [e]


def param_runner(f_run, param_dict):

    rows = []
    # get the names and values of all parameters
    p_names, p_values = param_dict.keys(), param_dict.values()

    # turn each value into a tuple with its index
    p_values = [[(i, v) for i, v in enumerate(p_v)] for p_v in p_values]

    # get cartesian product of all params ie p_sets = [ ((0,p1[0]),(0,p2[0])... ), ((1,p1[1]),(0,p2[0]) ... )...]
    p_sets = itertools.product(*p_values)

    bad_types = [np.ndarray, types.FunctionType, tuple]

    # iterate through params and record results into param indexed matrix
    for p_set in list(p_sets):
        kw_args = {n: v[1] for n, v in zip(p_names, p_set)}
        result = f_run(**kw_args)
        # convert numpy params to strings
        kw_args = {k: str(v) if type(
            v) in bad_types else v for k, v in kw_args.items()}
        row = {**kw_args, **result}
        rows.append(row)

    return pd.DataFrame(rows)


def plot_dataframe(df, y_cols, x_col=None, line_cols=[], title_cols=[], aggregate='error', val_styles='', fix_yim=False, legend=True, ax_size=(8, 6), show=True):
    """ Add two arguments
    Arguments:
        title_cols: groups data into unique combinations of each title col, each combo the becomes one subplot
        line_cols: groups each title_col into unique combinations of each line colm, each combo the becomes one line on a subplot
        y_cols: the key(s) in the dataframe which contain the series to be plotted, if multiple are given then each is plotted seperatly
        x_col: the key which will be used as x series for all y_cols
        fix_yim: whether to use the same ylim on all of the subplots, calculated via min max for all lines
        ax_size: size of each subplot - (X, Y)
        val_styles: styles passed along the plt.plot corresponding to each y_col
        aggregate: after grouping by title/line, there may still be multiple rows for each unique combination
                   we produce a single series from each set of rows via this aggreatition method
        show: whether to call plt.show() at the end 
    """

    # set up
    y_cols = arr_if_scalar(y_cols)
    val_styles = arr_if_scalar(val_styles) * len(y_cols)

    # helper function to deal with empty array grouping and to auto make labels
    def group_by(dataframe, cols):
        if len(cols) > 0:
            grouped = dataframe.groupby(cols)
            for (vals, group) in grouped:
                vals = arr_if_scalar(vals)
                # build label by group vals
                label = ', '.join([f'{k}={v}' for k, v in zip(cols, vals)])
                yield label, group
        else:
            yield '', dataframe

    # makes first title col vary by row
    d1, d2 = 1, 1
    for i, t in enumerate(title_cols):
        n = len(pd.unique(df[t]))
        if i == 0:
            d1 *= n
        else:
            d2 *= n
    fig, axarr = make_subplt_arr(d1, d2)
    fig.set_size_inches(d2*ax_size[0], d1*ax_size[1])

    # set super title
    val_col_str = ' & '.join(y_cols)
    size = 'xx-large'
    if len(axarr) < 5:
        size = 'x-large'
    if len(axarr) == 1:
        size = 'medium'

    fig.suptitle(val_col_str, weight='bold', size=size, y=0.94)

    mins, maxs = [], []
    for i, (title_str, title_group) in enumerate(group_by(df, title_cols)):

        ax = axarr[i]
        ax.set_title(f'{title_str}')

        for j, val_col in enumerate(y_cols):

            for (label_str, line_group) in group_by(title_group, line_cols):

                # get ys
                ys = []
                for i, r in line_group.iterrows():
                    ys.append(r.get(val_col))

                # get x
                if x_col is None:
                    if aggregate == 'plot':
                        x = np.arange(len(ys))
                    else:
                        # set x based on size val_col
                        x = np.arange(len(ys[0]))
                else:
                    x = line_group.iloc[0][x_col]
                    ax.set_xlabel(x_col)

                # compute errs
                if aggregate == 'error':
                    y, e = np.mean(ys, axis=0), np.std(ys, axis=0)
                    ax.fill_between(x, y-e, y+e, alpha=0.20)
                    maxs.append(np.max(y+e))
                    mins.append(np.min(y-e))

                elif aggregate == 'last':
                    # take last
                    y = ys[-1]
                elif aggregate == 'plot':
                    y = ys

                mins.append(np.min(y))
                maxs.append(np.max(y))

                if len(y_cols) > 1:
                    label_str = f'{label_str} {val_col}'

                ax.plot(x, y, label=label_str, *val_styles[j])

            if legend and len(line_cols) != 0:
                ax.legend()

    if fix_yim:
        [ax.set_ylim([min(mins), max(maxs)]) for ax in axarr]

    if show:
        plt.show()

    else:
        return fig, axarr


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


def convert_dtms_windowms_to_factor_dim(dt, window, TIME=1.0, LEN=10000):
    orignal_dt_ms = TIME/LEN * 1000
    factor = int(dt/orignal_dt_ms)
    dim = int(window / dt)
    return factor, dim


def count_phases(sig):
    N = len(sig)
    phases = 1
    for i in range(N-1):
        s1 = 1 if sig[i] >= 0 else -1
        s2 = 1 if sig[i+1] >= 0 else -1
        if s1 != s2:
            phases += 1
    return phases
