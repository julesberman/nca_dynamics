from cProfile import label
import scipy
import numpy as np
from fitters import *
from results import Results
from tools import *
from matplotlib.pyplot import figure
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
                      'figure.dpi': 80,
                      'figure.figsize': FIG_SIZE,
                      })


def plt_errors(x, y, e, *args, **kwargs):
    plt.plot(x, y, *args, **kwargs)
    plt.fill_between(x, y-e, y+e, alpha=0.2)


def plot_beta_errs(results, train=True, test=True, plot_opt=True, plot_errs=True, title=''):

    if train:
        for i, result in enumerate(results):
            betas, errs, l = result.betas, result.get_train_errs(), result.name
            if plot_errs:
                plt_errors(betas, *errs, '.-', label=l)
            else:
                plt.plot(betas, errs[0], label=l)
            if plot_opt:
                b_i = result.get_opt_beta_i()
                plt.scatter(betas[b_i], errs[0][b_i],
                            c='yellow', marker='*', zorder=999)

        plt.legend()
        plt.xlabel('Beta')
        plt.title(f'{title} Train Error')

        plt.show()

    if test:
        for i, result in enumerate(results):
            betas, errs, l = result.betas, result.get_test_errs(), result.name
            if plot_errs:
                plt_errors(betas, *errs, '.-', label=l, zorder=1)
            else:
                plt.plot(betas, errs[0], label=l)
            if plot_opt:
                b_i = result.get_opt_beta_i()
                plt.scatter(betas[b_i], errs[0][b_i],
                            c='yellow', marker='*', zorder=999)

        plt.legend()
        plt.xlabel('Beta')
        plt.title(f'{title} Test Error')
        plt.show()


def plot_params(params, title=''):
    for i, p in enumerate(params):
        if type(p[0][0]) is not np.ndarray:
            dist = p.flatten()
            plt.hist(dist, bins=len(dist)//100)
            plt.title(f'{title} Param {i}')
            plt.show()


def plot_filter(filter_mean_std, beta, title=''):
    figure(figsize=(8, 8))
    d = len(filter_mean_std[0])
    plt_errors(np.linspace(0, d, d), *filter_mean_std)
    plt.title(f'{title} Filter for beta={round(beta,3)}')
    plt.show()


def plot_spectrum(avg_spectrum, title=''):
    w, vl, vr = avg_spectrum
    d_ran = np.arange(0, len(w[0]), 1)
    plt_errors(d_ran, *w)
    plt.title('Eigenvalues of A')
    plt.show()
    plot_eigs(d_ran, vl[0], stds=vl[1],
              title=f'{title} Left Eig Vec of A')
    plot_eigs(d_ran, vr[0], stds=vr[1],
              title=f'{title} Right Eig Vec of A')


def plot_eigs(x, eigs, stds=None, title='', num=None):
    if num is not None:
        eigs = eigs[:num]
    num = len(eigs)
    r = math.ceil(math.sqrt(num))
    fig, axarr = plt.subplots(r, r)
    fig.suptitle(title)
    fig.set_size_inches(r*5, r*5)
    axs1 = [item for sublist in axarr for item in sublist]
    ylims = [np.min(eigs), np.max(eigs)]
    for i, e in list(enumerate(eigs)):
        if stds is not None:
            axs1[i].errorbar(x, e, stds[i])
        else:
            axs1[i].plot(x, e)
        axs1[i].set_ylim(ylims)

    plt.show()


def plot_self_corr(X, dim, title='', log_scale=True):
    X0 = build_hankel(X, dim)
    # X_corr = (X0 @ X0.T) / X0.shape[1]
    X_corr = np.corrcoef(X0)
    # X_corr /= np.sqrt(np.diag(X_corr))

    fig = figure(figsize=(8, 8))
    ax = sns.heatmap(X_corr, cmap="YlGnBu", square=True)
    ax.set_title(f'{title} X correlation')
    plt.show()

    w, vl, vr = scipy.linalg.eig(X_corr, left=True, right=True)
    d_ran = np.arange(0, len(w), 1)
    if log_scale:
        plt.semilogy(d_ran, w.real)
    else:
        plt.plot(d_ran, w.real)
    plt.title('Eigenvalues of correlation of X')
    plt.show()
    # num = 9
    # plot_eigs(d_ran, vl,
    #           title=f'{title} {num} Left EVec of X @ X.T', num=num)
    # plot_eigs(d_ran, vr,
    #           title=f'{title} {num} Right EVec of X @ X.T', num=num)
