# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy
import scipy.linalg

# %%
Exps = np.sort(np.array([2, -0.5, -1, -2, -3]))[::-1]
# Exps = np.sort(np.array([1.6, 1.1, 0.6, 0.1, -0.4, -1]))[::-1]

N_max = 13
Ns = np.arange(3, N_max + 1, 2)
num_runs = 20
T = np.int(2 * N_max)
shift = 3

time = np.linspace(0, T * 0.1, T + N_max)

noises = [
    0.000001,
    0.00001,
    0.0001,
    0.001,
    0.01,
    0.1,
]

data = {(N, noise): {"eigvecs": [], "eigvals": []}
        for N in Ns for noise in noises}
datasvd = {
    (N, noise): {"lefts": [], "rights": [], "lambdas": []}
    for N in Ns
    for noise in noises
}


for run_ in range(num_runs):

    Inits = 0.2 + 0.5 * np.random.rand(*Exps.shape)
    Xseries = np.exp(time.reshape(-1, 1) * Exps.reshape(1, -1)) @ Inits

    for noise in noises:
        XseriesNoisy = Xseries + noise * np.random.randn(*Xseries.shape)

        for N in Ns:

            X = scipy.linalg.hankel(XseriesNoisy)[
                : N + shift, : -N - shift + 1]

            X0 = X[:-shift]
            Xp = X[shift:]

            X0Xp = X0 @ Xp.T / X0.shape[1]
            X0X0 = X0 @ X0.T / X0.shape[1]

            GEV_sol = scipy.linalg.eig(X0Xp, X0X0)
            le, la, ri = scipy.linalg.svd(scipy.linalg.inv(X0X0) @ X0Xp)

            evals = np.real_if_close(GEV_sol[0])
            evecs = np.real_if_close(GEV_sol[1])

            sort_order = np.argsort(evals)[::-1]
            data[(N, noise)]["eigvals"].append(evals[sort_order])
            data[(N, noise)]["eigvecs"].append(evecs[:, sort_order])

            for i in range(len(ri)):
                si = np.sign(ri[i, -1])
                ri[i] = ri[i] * si
                le[:, i] = le[:, i] * si

            datasvd[(N, noise)]["lefts"].append(le)
            datasvd[(N, noise)]["lambdas"].append(la)
            datasvd[(N, noise)]["rights"].append(ri)

for N in Ns:
    for noise in noises:
        evals = np.array(data[(N, noise)]["eigvals"])
        data[(N, noise)]["eigvalmean"], data[(N, noise)]["eigvalstd"] = (
            np.mean(evals, 0),
            np.std(evals, 0),
        )
        evecs = np.array(data[(N, noise)]["eigvecs"])
        for r in range(num_runs):
            for i in range(evecs[r].shape[1]):
                evecs[r][:, i] *= np.sign(np.real(evecs[r][-1:, i]))

        data[(N, noise)]["eigvecs"] = evecs

        data[(N, noise)]["eigvecmean"], data[(N, noise)]["eigvecstd"] = (
            np.mean(evecs, 0),
            np.std(evecs, 0),
        )

        sort_order = np.argsort(data[(N, noise)]["eigvalmean"])[::-1]
        data[(N, noise)]["eigvalmeansrtd"] = data[(
            N, noise)]["eigvalmean"][sort_order]
        data[(N, noise)]["eigvalstdsrtd"] = data[(
            N, noise)]["eigvalstd"][sort_order]
        data[(N, noise)]["eigvecmeansrtd"] = data[(N, noise)]["eigvecmean"][
            :, sort_order
        ]
        data[(N, noise)]["eigvecmeansrtd"] = data[(N, noise)]["eigvecmean"][
            :, sort_order
        ]
        data[(N, noise)]["eigvecstdsrtd"] = data[(
            N, noise)]["eigvecstd"][:, sort_order]

        lambdas = np.array(datasvd[(N, noise)]["lambdas"])
        rights = np.array(datasvd[(N, noise)]["rights"])
        lefts = np.array(datasvd[(N, noise)]["lefts"])

        datasvd[(N, noise)]["lambdasmean"], datasvd[(N, noise)]["lambdasstd"] = (
            np.mean(lambdas, 0),
            np.std(lambdas, 0),
        )

        datasvd[(N, noise)]["rightsmean"], datasvd[(N, noise)]["rightsstd"] = (
            np.mean(rights, 0),
            np.std(rights, 0),
        )

        datasvd[(N, noise)]["leftsmean"], datasvd[(N, noise)]["leftsstd"] = (
            np.mean(lefts, 0),
            np.std(lefts, 0),
        )


# %%
for noise in noises:
    [
        plt.errorbar(
            np.arange(1, N + 1) + 0.02 * (N - N_max / 2),
            np.real(data[(N, noise)]["eigvalmeansrtd"]),
            np.real(data[(N, noise)]["eigvalstdsrtd"]),
        )
        for N in Ns[::-1]
    ]

    plt.scatter(
        np.arange(1, len(Exps) + 1),
        np.exp(shift * Exps * (time[1] - time[0])),
        c="black",
        ls=":",
    )
    plt.title("Eigenvalue. Real part Noise = {}".format(noise))
    plt.show()

    [
        plt.errorbar(
            np.arange(1, N + 1) + 0.02 * (N - N_max / 2),
            np.imag(data[(N, noise)]["eigvalmeansrtd"]),
            np.imag(data[(N, noise)]["eigvalstdsrtd"]),
        )
        for N in Ns[::-1]
    ]

    plt.title("Eigenvalue. Imaginary part Noise = {}".format(noise))
    plt.show()

    [
        plt.errorbar(
            np.arange(1, N + 1) + 0.02 * (N - N_max / 2),
            np.real(data[(N, noise)]["eigvecmean"][:, 0])[::-1],
            np.real(data[(N, noise)]["eigvecstd"][:, 0][::-1]),
        )
        for N in Ns[::-1]
    ]
    plt.axhline(0, ls=":", c="black", lw=1)
    plt.xticks(np.arange(1, N + 1), -np.arange(1, N + 1))
    plt.title("Top Eigenvector. Real part Noise = {}".format(noise))
    plt.show()

    [
        plt.errorbar(
            np.arange(1, N + 1) + 0.02 * (N - N_max / 2),
            np.imag(data[(N, noise)]["eigvecmean"][:, 0])[::-1],
            np.imag(data[(N, noise)]["eigvecstd"][:, 0][::-1]),
        )
        for N in Ns[::-1]
    ]
    plt.axhline(0, ls=":", c="black", lw=1)
    plt.xticks(np.arange(1, N + 1), -np.arange(1, N + 1))
    plt.title("Top Eigenvector. Imaginary part Noise = {}".format(noise))
    plt.show()

    print("-" * 30)


# %%
for noise in noises:
    [
        plt.errorbar(
            np.arange(1, N + 1) + 0.02 * (N - N_max / 2),
            np.real(datasvd[(N, noise)]["lambdasmean"]),
            np.real(datasvd[(N, noise)]["lambdasstd"]),
        )
        for N in Ns[::-1]
    ]

    plt.scatter(
        np.arange(1, len(Exps) + 1),
        np.exp(shift * Exps * (time[1] - time[0])),
        c="black",
        ls=":",
    )
    plt.title("Singular value. Noise = {}".format(noise))
    plt.show()

    [
        plt.errorbar(
            np.arange(1, N + 1) + 0.02 * (N - N_max / 2),
            np.real(datasvd[(N, noise)]["rightsmean"][0])[::-1],
            np.real(datasvd[(N, noise)]["rightsstd"][0][::-1]),
        )
        for N in Ns[::-1]
    ]
    plt.axhline(0, ls=":", c="black", lw=1)
    plt.xticks(np.arange(1, N + 1), -np.arange(1, N + 1))
    plt.title("Top right singular vector. Noise = {}".format(noise))
    plt.show()

    [
        plt.errorbar(
            np.arange(1, N + 1) + 0.02 * (N - N_max / 2),
            np.real(datasvd[(N, noise)]["leftsmean"][:, 0])[::-1],
            np.real(datasvd[(N, noise)]["leftsstd"][:, 0][::-1]),
        )
        for N in Ns[::-1]
    ]
    plt.axhline(0, ls=":", c="black", lw=1)
    plt.xticks(np.arange(1, N + 1), -np.arange(1, N + 1))
    plt.title("Top left singular vector. Noise = {}".format(noise))
    plt.show()

    print("-" * 30)
# %%
# for noise in noises:
#     [
#         plt.errorbar(
#             np.arange(1, N + 1) + 0.02 * (N - N_max / 2),
#             np.real(data[(N, noise)]["eigvecmean"][:,0])[::-1],
#             np.real(data[(N, noise)]["eigvecstd"][:,0][::-1] ),
#         )
#         for N in Ns[::-1]
#     ]
#     plt.axhline(0,ls=':',c='black',lw=1)
#     plt.xticks(np.arange(1, N + 1),-np.arange(1, N + 1))
#     plt.title("Top Eigenvector. Real part Noise = {}".format(noise))
#     plt.show()

# [
#     plt.errorbar(
#         np.arange(1, N + 1) + 0.02 * (N - N_max / 2),
#         np.imag(data[(N, noise)]["eigvecmean"][:,0]),
#         np.imag(data[(N, noise)]["eigvecstd"][:,0] ),
#     )
#     for N in Ns[::-1]
# ]

# plt.title("Top Eigenvector. Imaginary part Noise = {}".format(noise))
# plt.show()

# %%
# Defining iterative SVD functions


def svdv1(A, eta, n=100, u=None, v=None, l=None):

    if type(u) == type(None):
        u = np.random.randn(A.shape[0])
    if type(v) == type(None):
        v = np.random.randn(A.shape[1])
    if type(l) == type(None):
        l = (u @ A @ v) / (v @ v)

    l_hist, v_hist, u_hist = [], [], []

    l_hist.append(l)
    v_hist.append(np.copy(v))
    u_hist.append(np.copy(u))

    for _ in range(n):

        v += eta * (u @ A - l * v)
        l = (u @ A @ v) / (v @ v)
        u += eta * (A @ v - u)

        l_hist.append(l)
        v_hist.append(np.copy(v))
        u_hist.append(np.copy(u))

    return u, l, v, np.array(u_hist), np.array(v_hist), np.array(l_hist)


def svdv2(A, eta, n=100, u=None, v=None, l=None):

    if type(v) == type(None):
        v = np.random.randn(A.shape[1])
    if type(u) == type(None):
        u = A @ v
    if type(l) == type(None):
        l = (u @ A @ v) / (v @ v)

    l_hist, v_hist, u_hist = [], [], []

    l_hist.append(l)
    v_hist.append(np.copy(v))
    u_hist.append(np.copy(u))

    for _ in range(n):

        v += eta * (u @ A - l * v)
        u = A @ v
        l = (u @ A @ v) / (v @ v)

        l_hist.append(l)
        v_hist.append(np.copy(v))
        u_hist.append(np.copy(u))

    return u, l, v, np.array(u_hist), np.array(v_hist), np.array(l_hist)


# %%
noise = 0.000001
N = 7

Inits = 0.2 + 0.5 * np.random.rand(*Exps.shape)
Xseries = np.exp(time.reshape(-1, 1) * Exps.reshape(1, -1)) @ Inits
XseriesNoisy = Xseries + noise * np.random.randn(*Xseries.shape)

X = scipy.linalg.hankel(XseriesNoisy)[: N + shift, : -N - shift + 1]

X0 = X[:-shift]
Xp = X[shift:]

X0Xp = X0 @ Xp.T / X0.shape[1]
X0X0 = X0 @ X0.T / X0.shape[1]

A_ = scipy.linalg.inv(X0X0) @ X0Xp

le, la, ri = scipy.linalg.svd(A_)

sign = np.sign(le[-1, 0])
le *= sign
ri *= sign

u_, l_, v_, u_hist_, v_hist_, l_hist_ = svdv1(A_, 0.01, 1000)
unorm_hist_ = np.sqrt((u_hist_ ** 2).sum(1))
vnorm_hist_ = np.sqrt((v_hist_ ** 2).sum(1))

sign = np.sign(v_[-1])
v_ *= sign
u_ *= sign

plt.errorbar(
    np.arange(1, N + 1) + 0.02 * (N - N_max / 2),
    np.real(datasvd[(N, noise)]["leftsmean"][:, 0])[::-1],
    np.real(datasvd[(N, noise)]["leftsstd"][:, 0][::-1]),
)

plt.axhline(0, ls=":", c="black", lw=1)
plt.xticks(np.arange(1, N + 1), -np.arange(1, N + 1))
plt.title("Top left singular vector. Noise = {}".format(noise))
plt.plot(np.arange(1, N + 1), le[:, 0][::-1], label="top left (solver)")
plt.plot(
    np.arange(1, N + 1),
    u_[::-1] / np.sqrt(u_ @ u_),
    label="top left (iterative)",
    color="tab:red",
)

plt.axhline(0, ls=":", c="black", lw=1)
plt.xticks(np.arange(1, N + 1), -np.arange(1, N + 1))
plt.legend()
plt.show()

plt.errorbar(
    np.arange(1, N + 1) + 0.02 * (N - N_max / 2),
    np.real(datasvd[(N, noise)]["rightsmean"][0])[::-1],
    np.real(datasvd[(N, noise)]["rightsstd"][0][::-1]),
)

plt.axhline(0, ls=":", c="black", lw=1)
plt.xticks(np.arange(1, N + 1), -np.arange(1, N + 1))
plt.title("Top right singular vector. Noise = {}".format(noise))
plt.plot(np.arange(1, N + 1), ri[0][::-1], label="top right (solver)")
plt.plot(
    np.arange(1, N + 1),
    v_[::-1] / np.sqrt(v_ @ v_),
    label="top right (iterative)",
    color="tab:red",
)

plt.axhline(0, ls=":", c="black", lw=1)
plt.xticks(np.arange(1, N + 1), -np.arange(1, N + 1))
plt.legend()
plt.show()


plt.axhline(la[0], label='solver')
plt.axhline(datasvd[(N, noise)]["lambdasmean"][0],
            c="black", linewidth=1, label='mean')
plt.axhline(
    datasvd[(N, noise)]["lambdasmean"][0] +
    datasvd[(N, noise)]["lambdasstd"][0],
    c="black",
    linewidth=1,
    ls="--",
)
plt.axhline(
    datasvd[(N, noise)]["lambdasmean"][0] -
    datasvd[(N, noise)]["lambdasstd"][0],
    c="black",
    linewidth=1,
    ls="--",
)
plt.plot(l_hist_ * vnorm_hist_ / unorm_hist_,
         color="tab:red", label='iterative')
plt.title('Top singular value')
plt.legend()
plt.show()


# %%
# Defining the online training function
def v1(a, b, eta, n=100, w=None, l=None):

    if type(w) == type(None):
        w = np.random.randn(a.shape[0])
    else:
        w = np.copy(w)
    if type(l) == type(None):
        l = (w @ a @ w) / (w @ b @ w)

    l_hist, w_hist = [], []

    l_hist.append(l)
    w_hist.append(np.copy(w))

    for _ in range(n):

        w += eta * (a - l * b) @ w
        l = w @ a @ w / (w @ b @ w)

        l_hist.append(l)
        w_hist.append(np.copy(w))

    return w, l, w_hist, l_hist


def v2(a, b, eta, n=100, w=None, l=None):

    if type(w) == type(None):
        w = np.random.randn(a.shape[0])
    else:
        w = np.copy(w)
    if type(l) == type(None):
        l = (w @ a @ w) / (w @ b @ w)

    l_hist, w_hist = [], []

    l_hist.append(l)
    w_hist.append(np.copy(w))

    for _ in range(n):

        delta_w = eta * (a - l * b) @ w

        w += delta_w
        l = 0.01 * (w @ b @ w)

        l_hist.append(l)
        w_hist.append(np.copy(w))

    return w, l, w_hist, l_hist


# %%

# Inits = 0.5 + 0.5 * np.random.rand(*Exps.shape)
# Exps = np.sort(np.array([2, -0.5, -1, -1.5, -2]))[::-1]
# Inits = [0.1, 0.5, 1.5, 2.5, 3.4]
Exps = np.sort(np.array([1.5, -2]))[::-1]
Inits = [0.2, 6]
T = 6

time = np.linspace(0, T * 0.15, T + N_max)
XseriesIndivs = np.exp(time.reshape(-1, 1) * Exps.reshape(1, -1))

noise = 1e-5

for shift in range(1, 6):

    eigruns = []
    topeigruns = []
    topeignormedruns = []

    for _ in range(1000):

        # XseriesIndivsNoisy = XseriesIndivs + NoiseSeries.reshape(-1,1) * np.array([0, 1])
        # NoiseSeries = Inits[1]*noise * np.random.randn(len(XseriesIndivs))*XseriesIndivs[:,1]
        NoiseSeries = Inits[1] * noise * np.random.randn(len(XseriesIndivs))

        XseriesNoisy = XseriesIndivs @ Inits + NoiseSeries

        trueeigs = np.exp(Exps * (time[1] - time[0]))

        N = 5

        X = scipy.linalg.hankel(XseriesNoisy)[: N + shift, : -N - shift + 1]

        X0 = X[:-shift]
        Xp = X[shift:]

        X0Xp = X0 @ Xp.T / X0.shape[1]
        X0X0 = X0 @ X0.T / X0.shape[1]

        a = X0Xp
        b = X0X0

        GEV_sol = scipy.linalg.eig(a, b)
        top_nums = np.argsort(np.real(GEV_sol[0]))[::-1]
        eigs = np.real_if_close(GEV_sol[0][top_nums])
        eig_vecs = np.real_if_close(GEV_sol[1].T[top_nums])
        topeig = np.real_if_close(eig_vecs[0])
        topeig /= np.sign(topeig[-1])
        seceig = np.real_if_close(eig_vecs[1])
        seceig /= np.sign(seceig[-1])
        topeignormed = topeig / topeig[-1]
        seceignormed = seceig / seceig[-1]

        eigruns.append(eigs)
        topeigruns.append(topeig)
        topeignormedruns.append(topeignormed)

    plt.errorbar(
        np.arange(0, N),
        np.real(np.mean(topeigruns, 0))[::-1],
        np.std(topeigruns, 0)[::-1],
    )
    plt.xticks(np.arange(0, N), -np.arange(0, N))
    plt.title("Top eigenvalue (shift = {})".format(shift))
    plt.show()

    plt.errorbar(
        np.arange(0, N),
        np.real(np.mean(topeignormedruns, 0))[::-1],
        np.std(topeignormedruns, 0)[::-1],
    )
    plt.xticks(np.arange(0, N), -np.arange(0, N))
    plt.title("Top eigenvalue first normalized (shift = {})".format(shift))
    plt.show()

    plt.imshow(np.abs(eig_vecs @ eig_vecs.T))
    plt.colorbar()
    plt.show()

    plt.scatter(
        np.arange(1, len(Exps) + 1), trueeigs ** shift, c="black", ls=":",
    )
    plt.title("Real parts of the eigenvalues (shift = {})".format(shift))
    # plt.plot(
    #     np.arange(1, N + 1),
    #     np.real(np.array(eigruns))[:100].T,
    #     c="tab:red",
    #     alpha=0.03,
    #     marker="o",
    #     ls="",
    #     ms=3,
    # )
    plt.errorbar(
        np.arange(1, N + 1), np.real(np.mean(eigruns, 0)
                                     ), np.std(np.real(eigruns), 0)
    )
    plt.show()
    print("real parts of EVs:", np.real(np.mean(eigruns, 0)))
    print("True EVs:", np.sort(np.real(trueeigs))[::-1] ** shift)

    plt.errorbar(
        np.arange(1, N + 1), np.imag(np.mean(eigruns, 0)
                                     ), np.std(np.imag(eigruns), 0)
    )
    plt.title("Imaginary parts of the eigenvalues (shift = {})".format(shift))
    plt.show()


plt.plot(XseriesIndivs * Inits, c="tab:blue", label="component")
plt.plot(XseriesNoisy, c="black", label="total")
plt.plot(NoiseSeries, label="noise", c="tab:red")
plt.title("Time series")
plt.legend()
plt.show()

# %%
# checking stability of analytic solution on v2 algorithm
topeigv2 = topeig * np.sqrt(eigs[0] / 0.01 / (topeig @ b @ topeig))
wv2, lv2, whv2, lhv2 = v2(
    a, b, eta=0.0005, n=1000, w=np.real_if_close(topeigv2), l=np.max(np.real(eigs))
)
whv2 = np.array(whv2)
plt.axhline(np.max(np.real(eigs)), c="black", lw=1)
plt.plot(lhv2)
plt.show()
[plt.axhline(_, c="black", lw=1) for _ in topeig]
plt.plot(whv2[:, ::-1] / wv2[-1])
plt.show()

# %%
# can we get there from scratch
wv2, lv2, whv2, lhv2 = v2(a, b, eta=0.008, n=3000)
whv2 = np.array(whv2)
print(
    "computed: {:.4g} , analytic: {:.4g} , difference: {:.4g}".format(
        lv2, np.max(np.real(eigs)), np.max(np.real(eigs)) - lv2
    )
)
print()
print("GEV LHS/RHS:", (a @ wv2) / (b @ wv2))
plt.axhline(np.max(np.real(eigs)), c="black", lw=1)
plt.plot(lhv2)
plt.show()
[plt.axhline(_, c="black", lw=1) for _ in topeig]
plt.plot(whv2[:, ::-1] / wv2[-1])
plt.show()


plt.errorbar(
    np.arange(0, N),
    np.real(np.mean(topeignormedruns, 0))[::-1],
    np.std(topeignormedruns, 0)[::-1],
    label="analytic",
)
plt.plot(wv2[::-1] / wv2[-1], label="v2 solution")
plt.title("Eigenvector")
plt.xticks(np.arange(0, N), -np.arange(0, N))
plt.legend()
plt.show()

print(
    "overlap with top eigenvectors:\n", np.abs(
        GEV_sol[1].T @ wv2 / np.linalg.norm(wv2))
)

# %%
# Running the above longer

wv2, lv2, whv2, lhv2 = v2(a, b, eta=0.012, n=10000000, w=whv2[-1], l=lv2)
whv2 = np.array(whv2)
print(
    "computed: {:.4g} , analytic: {:.4g} , difference: {:.4g}".format(
        lv2, np.max(np.real(eigs)), np.max(np.real(eigs)) - lv2
    )
)
print()
print("GEV LHS/RHS:", (a @ wv2) / (b @ wv2))
plt.axhline(np.max(np.real(eigs)), c="black", lw=1)
plt.plot(lhv2)
plt.show()
[plt.axhline(_, c="black", lw=1) for _ in topeig]
plt.plot(whv2[:, ::-1] / wv2[-1])
plt.show()


plt.errorbar(
    np.arange(0, N),
    np.real(np.mean(topeignormedruns, 0))[::-1],
    np.std(topeignormedruns, 0)[::-1],
    label="analytic",
)
plt.plot(wv2[::-1] / wv2[-1], label="v2 solution")
plt.title("Eigenvector")
plt.xticks(np.arange(0, N), -np.arange(0, N))
plt.legend()
plt.show()

print(
    "overlap with top eigenvectors:\n", np.abs(
        GEV_sol[1].T @ wv2 / np.linalg.norm(wv2))
)

# %%
# Running the above even longer

wv2, lv2, whv2, lhv2 = v2(a, b, eta=0.013, n=30000000, w=whv2[-1], l=lv2)
whv2 = np.array(whv2)
print(lv2, np.max(np.real(eigs)))
print()
print((a @ wv2) / (b @ wv2))
print()
print(wv2 / wv2[-1])
plt.axhline(np.max(np.real(eigs)), c="black", lw=1)
plt.plot(lhv2)
plt.show()
[plt.axhline(_, c="black", lw=1) for _ in topeig]
plt.plot(whv2[:, ::-1] / wv2[-1])
plt.show()


plt.errorbar(
    np.arange(0, N),
    np.real(np.mean(topeignormedruns, 0))[::-1],
    np.std(topeignormedruns, 0)[::-1],
    label="analytic",
)
plt.plot(wv2[::-1] / wv2[-1], label="v2 solution")
plt.legend()
plt.title("Eigenvector")
plt.show()

print(np.abs(GEV_sol[1].T @ wv2 / np.linalg.norm(wv2)))


# %%
# Running the above with longer LR

wv2, lv2, whv2, lhv2 = v2(a, b, eta=0.014, n=3000000, w=whv2[-1], l=lv2)
whv2 = np.array(whv2)
print(lv2, np.max(np.real(eigs)))
print()
print((a @ wv2) / (b @ wv2))
print()
print(wv2 / wv2[-1])
plt.axhline(np.max(np.real(eigs)), c="black", lw=1)
plt.plot(lhv2)
plt.show()
[plt.axhline(_, c="black", lw=1) for _ in topeig]
plt.plot(whv2[:, ::-1] / wv2[-1])
plt.show()


plt.errorbar(
    np.arange(0, N),
    np.real(np.mean(topeignormedruns, 0))[::-1],
    np.std(topeignormedruns, 0)[::-1],
    label="analytic",
)
plt.plot(wv2[::-1] / wv2[-1], label="v2 solution")
plt.legend()
plt.title("Eigenvector")
plt.show()

print(np.abs(GEV_sol[1].T @ wv2 / np.linalg.norm(wv2)))


# %%

# Inits = 0.5 + 0.5 * np.random.rand(*Exps.shape)
Exps = np.sort(np.array([1, 0, 0, 0, 0]))[::-1]
Inits = [0.2, 0, 0, 0, 0]
T = 10

time = np.linspace(0, T * 0.2, T + N_max)
Xseries = np.exp(time.reshape(-1, 1) * Exps.reshape(1, -1)) @ Inits

noise = 1e-2

for _ in range(5):

    XseriesNoisy = Xseries + noise * np.random.randn(*Xseries.shape)

    trueeigs = np.exp(Exps * (time[1] - time[0]))

    N = 5

    X = scipy.linalg.hankel(XseriesNoisy)[: N + 1, :-N]

    X0 = X[:-1]
    Xp = X[1:]

    X0Xp = X0 @ Xp.T / X0.shape[1]
    X0X0 = X0 @ X0.T / X0.shape[1]

    a = X0Xp
    b = X0X0

    GEV_sol = scipy.linalg.eig(a, b)
    eigs = np.sort(np.real_if_close(GEV_sol[0][::-1]))[::-1]
    top_nums = np.argsort(np.real(GEV_sol[0]))[::-1]
    topeig = np.real_if_close(
        GEV_sol[1][:, top_nums[0]] / GEV_sol[1][-1, top_nums[0]])
    seceig = np.real_if_close(
        GEV_sol[1][:, top_nums[1]] / GEV_sol[1][-1, top_nums[1]])
    topeignormed = np.real_if_close(GEV_sol[1][:, top_nums[0]])
    seceignormed = np.real_if_close(GEV_sol[1][:, top_nums[1]])

    plt.plot(topeig[::-1])
    plt.xticks(np.arange(0, N), -np.arange(0, N))
    plt.title("Top eigenvalue")

plt.show()

plt.imshow(np.abs(GEV_sol[1].T @ GEV_sol[1]))
plt.colorbar()
plt.show()


plt.plot(XseriesNoisy)
plt.title("Time series")
plt.show()

plt.scatter(
    np.arange(1, len(Exps) + 1), trueeigs, c="black", ls=":",
)
plt.scatter(np.arange(1, len(eigs) + 1), np.real(eigs))
plt.title("Real parts of the eigenvalues")
plt.show()
print("real parts of EVs:", np.sort(np.real(eigs))[::-1])

plt.scatter(np.arange(1, len(eigs) + 1), np.imag(GEV_sol[0]))
plt.title("Imaginary parts of the eigenvalues")
plt.show()


# %%
# checking stability of analytic solution on v2 algorithm
topeigv2 = topeig * np.sqrt(eigs[0] / 0.01 / (topeig @ b @ topeig))
wv2, lv2, whv2, lhv2 = v2(
    a, b, eta=0.001, n=1000, w=np.real_if_close(topeigv2), l=np.max(np.real(eigs))
)
whv2 = np.array(whv2)
plt.axhline(np.max(np.real(eigs)), c="black", lw=1)
plt.plot(lhv2)
plt.show()
[plt.axhline(_, c="black", lw=1) for _ in topeig]
plt.plot(whv2[:, ::-1] / wv2[-1])
plt.show()

# %%
# can we get there from scratch
wv2, lv2, whv2, lhv2 = v2(a, b, eta=0.01, n=3000)
whv2 = np.array(whv2)
print(
    "computed: {:.4g} , analytic: {:.4g} , difference: {:.4g}".format(
        lv2, np.max(np.real(eigs)), np.max(np.real(eigs)) - lv2
    )
)
print()
print("GEV LHS/RHS:", (a @ wv2) / (b @ wv2))
plt.axhline(np.max(np.real(eigs)), c="black", lw=1)
plt.plot(lhv2)
plt.show()
[plt.axhline(_, c="black", lw=1) for _ in topeig]
plt.plot(whv2[:, ::-1] / wv2[-1])
plt.show()


plt.plot(topeig[::-1] / topeig[-1], label="analytic")
plt.plot(wv2[::-1] / wv2[-1], label="v2 solution")
plt.title("Eigenvector")
plt.xticks(np.arange(0, N), -np.arange(0, N))
plt.legend()
plt.show()

print(
    "overlap with top eigenvectors:\n", np.abs(
        GEV_sol[1].T @ wv2 / np.linalg.norm(wv2))
)

# %%
# Running the above longer

wv2, lv2, whv2, lhv2 = v2(a, b, eta=0.013, n=10000000, w=whv2[-1], l=lv2)
whv2 = np.array(whv2)
print(
    "computed: {:.4g} , analytic: {:.4g} , difference: {:.4g}".format(
        lv2, np.max(np.real(eigs)), np.max(np.real(eigs)) - lv2
    )
)
print()
print("GEV LHS/RHS:", (a @ wv2) / (b @ wv2))
plt.axhline(np.max(np.real(eigs)), c="black", lw=1)
plt.plot(lhv2)
plt.show()
[plt.axhline(_, c="black", lw=1) for _ in topeig]
plt.plot(whv2[:, ::-1] / wv2[-1])
plt.show()


plt.plot(topeig[::-1] / topeig[-1], label="analytic")
plt.plot(wv2[::-1] / wv2[-1], label="v2 solution")
plt.title("Eigenvector")
plt.xticks(np.arange(0, N), -np.arange(0, N))
plt.legend()
plt.show()

print(
    "overlap with top eigenvectors:\n", np.abs(
        GEV_sol[1].T @ wv2 / np.linalg.norm(wv2))
)

# %%
# Running the above even longer

wv2, lv2, whv2, lhv2 = v2(a, b, eta=0.01, n=30000000, w=whv2[-1], l=lv2)
whv2 = np.array(whv2)
print(lv2, np.max(np.real(eigs)))
print()
print((a @ wv2) / (b @ wv2))
print()
print(wv2 / wv2[-1])
plt.axhline(np.max(np.real(eigs)), c="black", lw=1)
plt.plot(lhv2)
plt.show()
[plt.axhline(_, c="black", lw=1) for _ in topeig]
plt.plot(whv2[:, ::-1] / wv2[-1])
plt.show()

plt.plot(topeig[::-1] / topeig[-1], label="analytic")
plt.plot(wv2[::-1] / wv2[-1], label="v2 solution")
plt.legend()
plt.title("Eigenvector")
plt.show()

print(np.abs(GEV_sol[1].T @ wv2 / np.linalg.norm(wv2)))

# %%
# checking stability of analytic solution on v1 algorithm
topeigv1 = topeig
wv1, lv1, whv1, lhv1 = v1(
    a, b, eta=0.01, n=1000000, w=topeigv1, l=np.max(np.real(eigs))
)
whv1 = np.array(whv1)
print(lv1, np.max(np.real(eigs)))
print()
print((a @ wv1) / (b @ wv1))
print()
print(wv1 / wv1[-1])
plt.axhline(np.max(np.real(eigs)), c="black", lw=1)
plt.plot(lhv1)
plt.show()
[plt.axhline(_, c="black", lw=1) for _ in topeig]
plt.plot(whv1[:, ::-1] / wv1[-1])
plt.show()

# %%

wv1, lv1, whv1, lhv1 = v1(a, b, eta=0.1, n=10000)
whv1 = np.array(whv1)
print(lv1, np.max(trueeigs))
print()
print((a @ wv1) / (b @ wv1))
print()
print(wv1 / wv1[-1])
plt.axhline(np.max(trueeigs), c="black", lw=1)
plt.plot(lhv1)
plt.show()
[plt.axhline(_, c="black", lw=1) for _ in topeig]
plt.plot(whv1[:, ::-1] / wv1[-1])
plt.show()
# %%
wv1, lv1, whv1, lhv1 = v1(a, b, eta=0.13, n=100000, w=wv1, l=lv1)
whv1 = np.array(whv1)
print(lv1, np.max(trueeigs), np.max(np.real(eigs)))
print()
print((a @ wv1) / (b @ wv1))
print()
print(wv1 / wv1[-1])
plt.axhline(np.max(np.real(eigs)), c="black", lw=1, ls=":")
plt.axhline(np.max(trueeigs), c="black", lw=1)
plt.plot(lhv1)
plt.show()
[plt.axhline(_, c="black", lw=1) for _ in topeig]
plt.plot(whv1[:, ::-1] / wv1[-1])
plt.show()

plt.plot(topeig[::-1] / topeig[-1], label="analytic")
plt.plot(wv1[::-1] / wv1[-1], label="v2 solution")
plt.title("Eigenvector")
plt.show()
# %%

# %%

# %%
