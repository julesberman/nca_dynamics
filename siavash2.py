# %%

from matplotlib import tight_layout
import numpy as np, matplotlib.pyplot as plt, scipy, scipy.linalg, pandas as pd, seaborn as sns


def nan_argsort(a):
    temp = a.copy()
    temp[np.isnan(a)] = -np.inf
    return temp.argsort()


Exps = np.sort(np.array([1.5, -0.15, -0.8]))

N_max = 10
Ns = np.arange(3, N_max + 1, 3)
num_runs = 20
T = np.int(2 * N_max)
shifts = np.arange(1, 11, 3)
shift_max = max(shifts)

time = np.arange(0, 2.55, 0.1)


noise = 0
results_eval = {
    "eval_r": [],
    "eval_im": [],
    "noise": [],
    "order": [],
    "order_num": [],
    "N": [],
    "run": [],
}

order_dict = {0: "1st", 1: "2nd", 2: "3rd"}
for i in range(3, 20):
    order_dict[i] = str(i + 1) + "th"

results_evec = {
    "evec_r": [],
    "evec_im": [],
    "noise": [],
    "component": [],
    "N": [],
    "run": [],
    "X_total": [],
    "X_target": [],
}

noises = [0.0000001, 0.0001, 0.0003, 0.001, 0.004, 0.01, 0.03]
Ns = [2, 3, 4, 6, 10]
runs = range(50)
for N in Ns:
    for noise in noises:
        for run in runs:

            Inits = np.array([0.01, 0.2, 1])[::-1] * (1 + 0 * 0.1 * np.random.randn(3))
            Xseries = np.exp(time.reshape(-1, 1) * Exps.reshape(1, -1)) * Inits.reshape(
                1, -1
            )

            XseriesNoisy = Xseries * (1 + noise * np.random.randn(*Xseries.shape))
            XsereisTotal = XseriesNoisy.sum(1)

            shift = 1
            X = scipy.linalg.hankel(XsereisTotal)[: N + shift, : -N - shift + 1]

            X0 = X[:-shift]
            Xp = X[shift:]
            X0Xp = X0 @ Xp.T / X0.shape[1]
            X0X0 = X0 @ X0.T / X0.shape[1]

            GEV_sol = scipy.linalg.eig(X0Xp, X0X0)
            evals = np.real_if_close(GEV_sol[0])
            evecs = np.real_if_close(GEV_sol[1])

            sort_ord = nan_argsort(np.real(evals))[::-1]
            for i, ord in enumerate(sort_ord):
                results_eval["eval_r"].append(np.real(evals[ord]))
                results_eval["eval_im"].append(np.imag(evals[ord]))
                results_eval["noise"].append(noise)
                results_eval["order"].append(order_dict[i])
                results_eval["order_num"].append(i)
                results_eval["N"].append(N)
                results_eval["run"].append(run)

            sign = np.sign(np.real(evecs[-1, sort_ord[0]]))
            for i, el in enumerate(evecs[:, sort_ord[0]]):
                results_evec["noise"].append(noise)
                results_evec["component"].append(-N + i)
                results_evec["N"].append(N)
                results_evec["evec_r"].append(np.real(sign * el))
                results_evec["evec_im"].append(np.imag(sign * el))
                results_evec["run"].append(run)
                results_evec["X_total"].append(XsereisTotal)
                results_evec["X_target"].append(XseriesNoisy[:, -1])


sns.set_theme(style="white")
sns.lineplot(x=time, y=XsereisTotal, label="$x(t)$", color="black")
palette = sns.color_palette("colorblind", 3)
[
    sns.lineplot(
        x=time,
        y=XseriesNoisy[:, i],
        color=palette[i],
        lw=1.5,
        label="$x_{}(t)$".format(i),
    )
    for i in range(3)
]
plt.xlabel("time (s)", fontsize=15)
plt.ylabel("series", fontsize=15)
plt.legend(fontsize=14, loc=1, ncol=2)
plt.savefig("series.pdf")
plt.show()

results_eval_df = pd.DataFrame(results_eval)
results_eval_df = results_eval_df[
    ~results_eval_df.isin([np.nan, np.inf, -np.inf]).any(1)
]
results_evec_df = pd.DataFrame(results_evec)

# %%
fig, axs = plt.subplots(
    3, 3, figsize=(12, 9), sharex=True, sharey=True, tight_layout=True
)
axs = axs.reshape(-1)

for i, noise in enumerate(noises):
    ax = axs[i]
    sns.set_theme(style="whitegrid")
    ax.scatter(
        range(len(Exps)),
        np.exp(Exps * 0.1)[::-1],
        marker="_",
        s=1300,
        c="tab:orange",
        alpha=0.7,
    )
    sns.violinplot(
        data=results_eval_df.query("noise=={} and order_num<6".format(noise)),
        y="eval_r",
        hue="N",
        x="order",
        # hue_order=["train", "valid"],
        # split=True,
        # inner="quartile",
        palette="Set2",
        ax=ax,
        legend=False,
    )
    ax.legend([], [], frameon=False)
    ax.set_xlabel("")
    if i in [0, 3, 6]:
        ax.set_ylabel("Eigenvalue")
    else:
        ax.set_ylabel("")
    ax.set_title("noise = {}".format(noise))
    if i in [6, 7, 8]:
        ax.set_xlabel("order")
    ax.set_ylim(-0.5, 1.5)
axs[0].legend(loc=3, ncol=2)
plt.savefig("eigval_comp.pdf")
plt.show()

# %%
fig, axs = plt.subplots(2, 3, figsize=(14, 6.5), sharex=True, sharey=True)
axs = axs.reshape(-1)

for i, noise in enumerate(noises[:-3]):
    ax = axs[i]
    sns.set_theme(style="whitegrid")
    sns.pointplot(
        data=results_evec_df.query("noise=={}".format(noise)),
        y="evec_r",
        hue="N",
        x="component",
        # hue_order=["train", "valid"],
        # split=True,
        # inner="quartile",
        palette="Set2",
        ax=ax,
        legend=False,
    )
    ax.legend([], [], frameon=False)
    ax.set_xlabel("")
    if i in [0, 3, 6]:
        ax.set_ylabel("Top Eigenvector")
    else:
        ax.set_ylabel("")
    ax.set_title("noise = {}".format(noise))
    if i in [3, 4, 5]:
        ax.set_xlabel("lag component")
axs[0].legend(loc=3, ncol=2)
plt.savefig("eigvec_comp.pdf")
plt.show()

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), tight_layout=True)
axs = axs.T.reshape(-1)

axs[0].scatter(
    range(len(Exps)), np.exp(Exps * 0.1)[::-1], marker="x", s=400, c="tab:red",
)
sns.set_theme(style="whitegrid")
sns.pointplot(
    data=results_eval_df.query(
        "N==3 and (noise==0.0000001 or noise==0.0003 or noise==0.03 or noise==0.1)"
    ),
    y="eval_r",
    hue="noise",
    x="order",
    palette="mako",
    legend=False,
    ax=axs[0],
)
axs[0].set_title("Eigenvalues")
axs[0].set_ylabel("")
axs[0].set_xlabel("Eigenvalue order (largest first)")

sns.set_theme(style="whitegrid")
sns.pointplot(
    data=results_evec_df.query(
        "N==3 and (noise==0.01 or noise==0.0001 or noise==0.0003 or noise==0.002 or noise==0.1)"
    ),
    y="evec_r",
    hue="noise",
    x="component",
    palette="mako",
    legend=False,
    ax=axs[1],
)
axs[1].set_title("Top eigenvector")
axs[1].set_ylabel("")
axs[1].set_xlabel("lag component")
plt.savefig("N=3.pdf")
# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), tight_layout=True)
axs = axs.T.reshape(-1)

sns.set_theme(style="whitegrid")
axs[0].scatter(
    range(len(Exps)), np.exp(Exps * 0.1)[::-1], marker="x", s=400, c="tab:red",
)
sns.pointplot(
    data=results_eval_df.query(
        "N==6 and (noise==0.0000001 or noise==0.0003 or noise==0.03 or noise==0.1)"
    ),
    y="eval_r",
    hue="noise",
    x="order",
    palette="mako",
    legend=False,
    ax=axs[0],
)
axs[0].set_title("Eigenvalues")
axs[0].set_ylabel("")
axs[0].set_xlabel("Eigenvalue order (largest first)")

sns.set_theme(style="whitegrid")
sns.pointplot(
    data=results_evec_df.query(
        "N==6 and (noise==0.01 or noise==0.0001 or noise==0.0003 or noise==0.002 or noise==0.1)"
    ),
    y="evec_r",
    hue="noise",
    x="component",
    palette="mako",
    legend=False,
    ax=axs[1],
)
axs[1].set_title("Top eigenvector")
axs[1].set_ylabel("")
axs[1].set_xlabel("lag component")
plt.savefig("N=6.pdf")

# %%

Inits = np.array([0.01, 0.2, 1])[::-1] * (1 + 0 * 0.1 * np.random.randn(3))
Xseries = np.exp(time.reshape(-1, 1) * Exps.reshape(1, -1)) * Inits.reshape(1, -1)
XseriesNoisy = Xseries
XsereisTotal = XseriesNoisy.sum(1)

proj_results = {"run": [], "N": [], "t": [], "x": [], "noise": []}

for N in Ns:
    X = scipy.linalg.hankel(XsereisTotal)[:N, : -N + 1]
    for noise in noises:
        for run in runs:
            evec = results_evec_df.query(
                "N=={} and noise=={} and run=={}".format(N, noise, run)
            )["evec_r"].to_numpy()
            proj_series = evec @ X
            proj_series = proj_series / proj_series[0] * Xseries[N - 1, -1]
            for i, el in enumerate(proj_series):
                proj_results["run"].append(run)
                proj_results["N"].append(N)
                proj_results["t"].append(time[N + i - 1])
                proj_results["x"].append(el)
                proj_results["noise"].append(noise)

proj_results_pd = pd.DataFrame(proj_results)
# %%
fig, axs = plt.subplots(
    2, 3, figsize=(12, 8), sharex=True, sharey=True, tight_layout=True
)
palette = sns.color_palette("mako_r", len(Ns))
axs = axs.reshape(-1)

for i, noise in enumerate([0.0000001, 0.0001, 0.001, 0.01, 0.03, 0.1]):
    ax = axs[i]
    sns.set_theme(style="whitegrid")
    ax.plot(time, Xseries[:, -1], color="red")
    sns.lineplot(
        data=proj_results_pd.query("noise=={}".format(noise)),
        y="x",
        hue="N",
        x="t",
        err_style="bars",
        palette=palette,
        style="N",
        ax=ax,
        alpha=0.9,
    )

    # ax.legend([], [], frameon=False)
    ax.set_xlabel("")
    if i in [0, 3, 6]:
        ax.set_ylabel("Projection")
    else:
        ax.set_ylabel("")
    ax.set_title("noise = {}".format(noise))
    if i in [6, 7, 8]:
        ax.set_xlabel("time")
    ax.set_ylim(-0.05, 0.49)
# axs[0].legend(loc=3, ncol=2)
plt.savefig("proj_comp.pdf")
plt.show()

# %%

Inits = np.array([0.01, 0.2, 1])[::-1] * (1 + 0 * 0.1 * np.random.randn(3))
Xseries = np.exp(time.reshape(-1, 1) * Exps.reshape(1, -1)) * Inits.reshape(1, -1)
XseriesNoisy = Xseries
XsereisTotal = XseriesNoisy.sum(1)

proj_results_noisy = {
    "run": [],
    "N": [],
    "t": [],
    "x": [],
    "noise": [],
    "target": [],
    "i": [],
}

for N in Ns:
    for noise in noises:
        for run in runs:
            XsereisTotal = results_evec_df.query(
                "N=={} and noise=={} and run=={}".format(N, noise, run)
            )["X_total"].iloc[0]
            Xsereistarget = results_evec_df.query(
                "N=={} and noise=={} and run=={}".format(N, noise, run)
            )["X_target"].iloc[0]
            X = scipy.linalg.hankel(XsereisTotal)[:N, : -N + 1]
            evec = results_evec_df.query(
                "N=={} and noise=={} and run=={}".format(N, noise, run)
            )["evec_r"].to_numpy()
            proj_series = evec @ X
            proj_series = proj_series / proj_series[0] * Xseries[N - 1, -1]
            for i, el in enumerate(proj_series):
                proj_results_noisy["run"].append(run)
                proj_results_noisy["N"].append(N)
                proj_results_noisy["t"].append(time[N + i - 1])
                proj_results_noisy["x"].append(el)
                proj_results_noisy["noise"].append(noise)
                proj_results_noisy["target"].append(Xsereistarget[N + i - 1])
                proj_results_noisy["i"].append(i)

proj_results_noisy_pd = pd.DataFrame(proj_results_noisy)

fig, axs = plt.subplots(
    2, 3, figsize=(12, 8), sharex=True, sharey=True, tight_layout=True
)
palette = sns.color_palette("mako_r", len(Ns))
axs = axs.reshape(-1)

for i, noise in enumerate([0.0000001, 0.0001, 0.001, 0.01, 0.03, 0.1]):
    ax = axs[i]
    sns.set_theme(style="whitegrid")
    ax.plot(time, Xseries[:, -1], color="red")
    sns.lineplot(
        data=proj_results_noisy_pd.query("noise=={}".format(noise)),
        y="target",
        x="t",
        err_style="bars",
        palette=palette,
        color="red",
        ax=ax,
    )
    sns.lineplot(
        data=proj_results_noisy_pd.query("noise=={}".format(noise)),
        y="x",
        hue="N",
        x="t",
        err_style="bars",
        palette=palette,
        style="N",
        ax=ax,
    )

    # ax.legend([], [], frameon=False)
    ax.set_xlabel("")
    if i in [0, 3, 6]:
        ax.set_ylabel("Projection")
    else:
        ax.set_ylabel("")
    ax.set_title("noise = {}".format(noise))
    if i in [6, 7, 8]:
        ax.set_xlabel("time")
    ax.set_ylim(-0.05, 0.49)
# axs[0].legend(loc=3, ncol=2)
plt.savefig("proj_noisy_comp.pdf")
plt.show()

# %%
fig, axs = plt.subplots(
    1, 4, figsize=(10, 3), sharex=True, sharey=True, tight_layout=True
)
palette = sns.color_palette("mako_r", len(Ns))
axs = axs.reshape(-1)

for i, noise in enumerate([0.001, 0.01, 0.03, 0.1]):

    XseriesNoisy = Xseries * (1 + noise * np.random.randn(*Xseries.shape))
    XsereisTotal = XseriesNoisy.sum(1)

    ax = axs[i]

    ax.plot(time, XsereisTotal, color="black")
    # ax.legend([], [], frameon=False)
    ax.set_xlabel("")
    if i in [0]:
        ax.set_ylabel("Time sereis")
    else:
        ax.set_ylabel("")
    ax.set_title("noise = {}".format(noise))
    # if i in [2, 3]:
    ax.set_xlabel("time")
    ax.plot(time, XsereisTotal, c="black")
# axs[0].legend(loc=3, ncol=2)
plt.savefig("total_series_noisy.pdf")
plt.show()

#%%

plt.axhline(0, color="tab:orange", lw=0.8)
sns.set_theme(style="whitegrid")
palette = sns.color_palette("mako", 3)
paper_rc = {"lines.linewidth": 1.2, "lines.markersize": 10}
sns.set_context("paper", rc=paper_rc)
sns.pointplot(
    data=results_evec_df[results_evec_df["N"].isin([3, 6, 10])].query("noise==0.0001"),
    y="evec_r",
    hue="N",
    x="component",
    palette=palette,
    legend=False,
    lw=0.5,
)
plt.xlabel("lag component", fontsize=14)
plt.ylabel("", fontsize=14)
plt.title("Top left Eigenvector", fontsize=14)
legend = plt.legend(ncol=3, title="lag vector length (n)", loc=3, fontsize=13)
plt.setp(legend.get_title(), fontsize=13)
plt.yticks([-0.6, -0.3, 0, 0.3, 0.6], fontsize=11)
plt.savefig("vslength.pdf")
# %%
sns.set_theme(style="white")
sns.lineplot(
    x=time, y=XsereisTotal, label="$x(t)$", color="black", lw=2,
)
palette = sns.color_palette("mako", 4)
paper_rc = {"lines.linewidth": 1.2, "lines.markersize": 10}
sns.set_context("paper", rc=paper_rc)
[
    sns.lineplot(
        x=time,
        y=XseriesNoisy[:, i],
        color=palette[i + 1],
        lw=1.5,
        label="$x_{}(t)$".format(i),
    )
    for i in range(3)
]
plt.xlabel("time (s)", fontsize=15)
plt.title("Series and constituents", fontsize=15)
legend = plt.legend(fontsize=13, loc=1, ncol=2)
plt.setp(legend.get_title(), fontsize=13)
plt.yticks([0, 0.3, 0.6, 0.9, 1.2], fontsize=11)
plt.grid(axis="y")
plt.savefig("series.pdf")
plt.show()
# %%

proj_results_noisy = {
    "run": [],
    "N": [],
    "t": [],
    "x": [],
    "noise": [],
    "target": [],
    "i": [],
}

for N in Ns:
    for noise in noises:
        for run in runs:
            XsereisTotal = results_evec_df.query(
                "N=={} and noise=={} and run=={}".format(N, noise, run)
            )["X_total"].iloc[0]
            Xsereistarget = results_evec_df.query(
                "N=={} and noise=={} and run=={}".format(N, noise, run)
            )["X_target"].iloc[0]
            X = scipy.linalg.hankel(XsereisTotal)[:N, : -N + 1]
            evec = results_evec_df.query(
                "N=={} and noise=={} and run=={}".format(N, noise, run)
            )["evec_r"].to_numpy()
            proj_series = evec @ X
            proj_series = proj_series / proj_series[0] * Xseries[N - 1, -1]
            for i, el in enumerate(proj_series):
                proj_results_noisy["run"].append(run)
                proj_results_noisy["N"].append(N)
                proj_results_noisy["t"].append(time[N + i - 1])
                proj_results_noisy["x"].append(el)
                proj_results_noisy["noise"].append(noise)
                proj_results_noisy["target"].append(Xsereistarget[N + i - 1])
                proj_results_noisy["i"].append(i)

proj_results_noisy_pd = pd.DataFrame(proj_results_noisy)

# %%
fig, axs = plt.subplots(
    1, 3, figsize=(11, 3.3), sharex=True, sharey=True, tight_layout=True
)
palette = sns.color_palette("mako", 3)
axs = axs.reshape(-1)

for i, noise in enumerate([0.0001, 0.004, 0.03]):
    ax = axs[i]
    sns.set_theme(style="white")
    # ax.plot(time, Xseries[:, -1], color="red")

    sns.lineplot(
        data=proj_results_noisy_pd[proj_results_noisy_pd["N"].isin([2, 4, 10])].query(
            "noise=={}".format(noise)
        ),
        y="x",
        hue="N",
        x="t",
        err_style="bars",
        palette=palette,
        # style="N",
        ax=ax,
    )
    sns.lineplot(
        data=proj_results_noisy_pd[proj_results_noisy_pd["N"].isin([2, 4, 10])].query(
            "noise=={}".format(noise)
        ),
        y="target",
        x="t",
        err_style="bars",
        palette=palette,
        color="tab:red",
        ax=ax,
        label="target",
        alpha=0.8,
    )
    ax.lines[-2].set_linestyle("--")

    ax.legend([], [], frameon=False)
    ax.set_xlabel("")
    if i in [0, 3, 6]:
        ax.set_ylabel("Projection")
    else:
        ax.set_ylabel("")
    ax.set_title("noise = {}".format(noise))
    ax.set_xlabel("time (s)")
    ax.set_ylim(-0.05, 0.49)
    ax.grid(axis="y")
axs[0].legend(ncol=2, title="lag vector length (n)", loc=2)
plt.yticks([0, 0.15, 0.3, 0.45])

plt.savefig("proj_noisy_comp.pdf")
plt.show()
# %%
# %%
