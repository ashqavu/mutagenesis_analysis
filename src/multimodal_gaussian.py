import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


def multimodal_gaussian_model(data, drug, ax=None, read_threshold=20, sigma_cutoff=4):
    if ax is None:
        ax = plt.gca()
    fitness_dfs = data.filter_fitness_read_noise(read_threshold=read_threshold)
    # * determine bins
    fitness_all = []
    for treatment in data.treatments:
        if "UT" in treatment:
            continue
        fitness_treatment = fitness_dfs[treatment]
        fitness_all.extend(fitness_treatment.drop(["*", "∅"], axis=1).values.flatten())
    bins = np.linspace(
        np.nanpercentile(fitness_all, 0.1), np.nanpercentile(fitness_all, 99.9), 40
    )

    fitness_df = fitness_dfs[drug]
    X = fitness_df.values.flatten()
    X = X[~np.isnan(X)]

    values_syn = fitness_df["∅"].values.flatten()
    values_stop = fitness_df["*"].values.flatten()

    y, x = np.histogram(X, bins=bins)

    sns.histplot(
        X, alpha=0.8, color="gray", label="all mutations", bins=bins, ax=ax, ec="white"
    )
    sns.histplot(
        values_syn,
        alpha=0.8,
        color="greenyellow",
        label="synonymous mutations",
        bins=bins,
        ax=ax,
        ec="white",
    )
    sns.histplot(
        values_stop,
        alpha=0.8,
        color="lightcoral",
        label="stop mutations",
        bins=bins,
        ax=ax,
        ec="white",
    )
    peaks, _ = find_peaks(y, height=50, threshold=5)

    try:
        y, x = np.histogram(X, bins=bins)

        x = (x[1:] + x[:-1]) / 2  # for len(x) == len(y)

        # x, y inputs can be lists or 1D numpy arrays

        def gauss(x, mu, sigma, A):
            return A * np.exp(-((x - mu) ** 2) / 2 / sigma**2)

        def trimodal(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3):
            return (
                gauss(x, mu1, sigma1, A1)
                + gauss(x, mu2, sigma2, A2)
                + gauss(x, mu3, sigma3, A3)
            )

        expected = (
            bins[peaks][0],
            0.2,
            0,
            bins[peaks][1],
            0.2,
            0,
            bins[peaks][2],
            0.2,
            0,
        )
        params, cov = curve_fit(trimodal, x, y, expected, maxfev=3000)
        sigma = np.sqrt(np.diag(cov))
        x_fit = np.linspace(x.min(), x.max(), 500)

        residuals = y - trimodal(x, *params)
        ss_res = np.sum(residuals**2)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_total)

        df_params = pd.DataFrame(
            data={"params": params, "sigma": sigma},
            index=trimodal.__code__.co_varnames[1:],
        )
        print(f"r^2\t{r_squared}")
        print(df_params)

        # plot combined...
        ax.plot(x_fit, trimodal(x_fit, *params), color="black", lw=1.25, label="model")
        # ...and individual Gauss curves
        ax.plot(
            x_fit,
            gauss(x_fit, *params[:3]),
            color="black",
            lw=1,
            ls="--",
            label="distribution 1",
        )
        ax.plot(
            x_fit,
            gauss(x_fit, *params[3:6]),
            color="black",
            lw=1,
            ls=":",
            label="distribution 2",
        )
        ax.plot(
            x_fit,
            gauss(x_fit, *params[6:]),
            color="black",
            lw=1,
            ls="-.",
            label="distribution 3",
        )

        central_mu_idx = np.abs(params[::3]).argmin()
        central_mu = params[central_mu_idx * 3]
        central_std = params[central_mu_idx * 3 + 1]

        left_cutoff = central_mu - (sigma_cutoff * central_std)
        right_cutoff = central_mu + (sigma_cutoff * central_std)

        ax.axvline(
            left_cutoff,
            ax.get_ylim()[0],
            color="royalblue",
            linestyle="dashed",
            lw=1,
            zorder=10,
            label=rf"-{sigma_cutoff}$\sigma$",
        )
        ax.axvline(
            right_cutoff,
            ax.get_ylim()[0],
            color="crimson",
            linestyle="dashed",
            lw=1,
            zorder=10,
            label=rf"{sigma_cutoff}$\sigma$",
        )
        ax.axvline(
            central_mu,
            ax.get_ylim()[0],
            color="gray",
            linestyle="dashed",
            lw=1,
            zorder=10,
            label=r"central $\mu$",
        )

    except IndexError:
        try:
            y, x = np.histogram(X, bins=bins)
            peaks, _ = find_peaks(y, height=50, threshold=5)

            x = (x[1:] + x[:-1]) / 2  # for len(x) == len(y)

            # x, y inputs can be lists or 1D numpy arrays

            def gauss(x, mu, sigma, A):
                return A * np.exp(-((x - mu) ** 2) / 2 / sigma**2)

            def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
                return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2)

            expected = (bins[peaks][0], 0.2, 0, bins[peaks][1], 0.2, 0)
            params, cov = curve_fit(bimodal, x, y, expected, maxfev=3000)
            sigma = np.sqrt(np.diag(cov))
            x_fit = np.linspace(x.min(), x.max(), 500)

            residuals = y - bimodal(x, *params)
            ss_res = np.sum(residuals**2)
            ss_total = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_total)

            df_params = pd.DataFrame(
                data={"params": params, "sigma": sigma},
                index=bimodal.__code__.co_varnames[1:],
            )
            print(f"r^2\t{r_squared}")
            print(df_params)

            # plot combined...
            ax.plot(
                x_fit, bimodal(x_fit, *params), color="black", lw=1.25, label="model"
            )
            # ...and individual Gauss curves
            ax.plot(
                x_fit,
                gauss(x_fit, *params[:3]),
                color="black",
                lw=1,
                ls="--",
                label="distribution 1",
            )
            ax.plot(
                x_fit,
                gauss(x_fit, *params[3:]),
                color="black",
                lw=1,
                ls=":",
                label="distribution 2",
            )

            central_mu_idx = np.abs(params[::3]).argmin()
            central_mu = params[central_mu_idx * 3]
            central_std = params[central_mu_idx * 3 + 1]

            left_cutoff = central_mu - (sigma_cutoff * central_std)
            right_cutoff = central_mu + (sigma_cutoff * central_std)

            ax.axvline(
                left_cutoff,
                ax.get_ylim()[0],
                color="royalblue",
                linestyle="dashed",
                lw=1,
                zorder=10,
                label=rf"-{sigma_cutoff}$\sigma$",
            )
            ax.axvline(
                right_cutoff,
                ax.get_ylim()[0],
                color="crimson",
                linestyle="dashed",
                lw=1,
                zorder=10,
                label=rf"{sigma_cutoff}$\sigma$",
            )
            ax.axvline(
                central_mu,
                ax.get_ylim()[0],
                color="gray",
                linestyle="dashed",
                lw=1,
                zorder=10,
                label=r"central $\mu$",
            )

        except IndexError:
            y, x = np.histogram(X, bins=bins)
            peaks, _ = find_peaks(y, height=50, threshold=5)

            # data generation
            np.random.seed(123)
            x = (x[1:] + x[:-1]) / 2  # for len(x) == len(y)

            expected = (bins[peaks][0], 0.2, 0)
            params, cov = curve_fit(gauss, x, y, expected, maxfev=3000)
            sigma = np.sqrt(np.diag(cov))
            x_fit = np.linspace(x.min(), x.max(), 500)

            residuals = y - gauss(x, *params)
            ss_res = np.sum(residuals**2)
            ss_total = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_total)

            df_params = pd.DataFrame(
                data={"params": params, "sigma": sigma},
                index=gauss.__code__.co_varnames[1:],
            )
            print(f"r^2\t{r_squared}")
            print(df_params)

            ax.plot(x_fit, gauss(x_fit, *params), color="black", lw=1.25, label="model")
            central_mu = params[0]
            central_std = params[1]

            left_cutoff = central_mu - (sigma_cutoff * central_std)
            right_cutoff = central_mu + (sigma_cutoff * central_std)

            ax.axvline(
                left_cutoff,
                ax.get_ylim()[0],
                color="royalblue",
                linestyle="dashed",
                lw=1,
                zorder=10,
                label=rf"-{sigma_cutoff}$\sigma$",
            )
            ax.axvline(
                right_cutoff,
                ax.get_ylim()[0],
                color="crimson",
                linestyle="dashed",
                lw=1,
                zorder=10,
                label=rf"{sigma_cutoff}$\sigma$",
            )
            ax.axvline(
                central_mu,
                ax.get_ylim()[0],
                color="gray",
                linestyle="dashed",
                lw=1,
                zorder=10,
                label=r"central $\mu$",
            )

    except RuntimeError:
        pass

    try:
        textstr = f"$r^2$={round(r_squared, 3)}"
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize="small",
            va="top",
            bbox={"fc": "white", "boxstyle": "round"},
        )
    except UnboundLocalError:
        pass

    ax.legend(fontsize="xx-small")
    ax.set_title(drug)
    ax.set_xlabel("relative fitness")

    print(f"Peaks: {np.take(bins, peaks)}")
    print()

    return ax


def gaussian_mixture_hist(
    data,
    drug,
    ax=None,
    read_threshold=20,
    sigma_cutoff=4,
    xlim=(-2, 2),
    all_peaks=False,
    n_components=3,
):
    if ax is None:
        ax = plt.gca()

    fitness_dfs = data.filter_fitness_read_noise(read_threshold=read_threshold)

    fitness_df = fitness_dfs[drug]
    X = fitness_df.values.flatten()
    X = X[~np.isnan(X)]
    X = np.expand_dims(X, 1)

    # * determine bins
    fitness_all = []
    for treatment in data.treatments:
        if "UT" in treatment:
            continue
        fitness_treatment = fitness_dfs[treatment]
        fitness_all.extend(fitness_treatment.drop(["*", "∅"], axis=1).values.flatten())
    bins = np.linspace(
        np.nanpercentile(fitness_all, 0.1), np.nanpercentile(fitness_all, 99.9), 30
    )

    model = GaussianMixture(n_components=n_components, covariance_type="full")
    model.fit(X)
    weights = model.weights_
    means = model.means_
    covars = model.covariances_

    x_axis = np.arange(xlim[0], xlim[1], 0.1)
    y_axes = [
        norm.pdf(x_axis, float(means[i][0]), np.sqrt(float(covars[i][0][0])))
        * weights[i]
        for i in range(len(means))
    ]

    sns.histplot(
        X,
        stat="density",
        bins=bins,
        ax=ax,
        fc="darkgray",
        label="all mutations",
        zorder=-100,
    )
    # bin_edges = np.histogram_bin_edges(X, bins="auto")

    # * stop mutations
    X_stop = fitness_df["*"].values.flatten()
    X_stop = X_stop[~np.isnan(X_stop)]
    X_stop = np.expand_dims(X_stop, 1)

    line_styles = [":", "dashed", "dashdot"]
    for i, y in enumerate(y_axes):
        sns.lineplot(
            x=x_axis,
            y=y,
            lw=1.5,
            ls=line_styles[i],
            color="black",
            label=f"distribution {i+1}",
            ax=ax,
        )
    y_combined = np.add.reduce(y_axes)
    sns.lineplot(x=x_axis, y=y_combined, lw=2, color="black", label="model", ax=ax)

    # * determine cutoffs significance
    # calculate standard deviation from covariances
    stdevs = np.sqrt(covars)

    if all_peaks:
        # calculate mus and cutoffs
        for i, (mu, std) in enumerate(list(zip(means, stdevs))):
            left_cutoff = mu - (sigma_cutoff * std)
            right_cutoff = mu + (sigma_cutoff * std)

            ax.axvline(
                mu,
                ax.get_ylim()[0],
                color="gray",
                linestyle=line_styles[i],
                lw=1.5,
                zorder=10,
                label=rf"$\mu_{i}$",
            )
            ax.axvline(
                left_cutoff,
                ax.get_ylim()[0],
                color="royalblue",
                linestyle=line_styles[i],
                lw=1.5,
                zorder=10,
                label=rf"-${sigma_cutoff}\sigma$",
            )
            ax.axvline(
                right_cutoff,
                ax.get_ylim()[0],
                color="crimson",
                linestyle=line_styles[i],
                lw=1.5,
                zorder=10,
                label=rf"${sigma_cutoff}\sigma$",
            )
    else:
        # * central peak
        central_mu_idx = np.abs(means).argmin()
        central_mu = means[central_mu_idx]
        central_stdev = stdevs[central_mu_idx]

        left_cutoff = central_mu - (sigma_cutoff * central_stdev)
        right_cutoff = central_mu + (sigma_cutoff * central_stdev)

        ax.axvline(
            central_mu,
            ax.get_ylim()[0],
            color="gray",
            linestyle="dashed",
            lw=1,
            zorder=10,
            label=r"central $\mu$",
        )
        ax.axvline(
            left_cutoff,
            ax.get_ylim()[0],
            color="royalblue",
            linestyle="dashed",
            lw=1,
            zorder=10,
            label=rf"-{sigma_cutoff}$\sigma$",
        )
        ax.axvline(
            right_cutoff,
            ax.get_ylim()[0],
            color="crimson",
            linestyle="dashed",
            lw=1,
            zorder=10,
            label=rf"{sigma_cutoff}$\sigma$",
        )

    # * figure aesthetics
    ax.get_legend().remove()
    ax.legend(
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        bbox_transform=ax.figure.transFigure,
        frameon=False,
    )
    ax.set_title(drug, pad=10)
    ax.set_xlabel("relative fitness")

    # * print parameters
    df = pd.DataFrame(columns=["mean", "covariance", "weight", "stdev"])
    for i, (mean, covar, weight, std) in enumerate(list(zip(means, covars, weights, stdevs))):
        df.loc[f"distribution {i+1}"] = [mean[0], covar[0][0], weight, std[0][0]]
    print(df)

    return ax, model


def build_gaussian_model_1d_mixture(df: pd.DataFrame, n_components: int = 3) -> tuple[float, float]:
    """
    Use relative fitness values of all mutations to build a Gaussian mixture model
    that will determine the 1-D bounds of significance for fitness effects

    Parameters
    ----------
    df : pd.DataFrame
        Relative fitness data for drug
    n_components : int, optional
        Number of components to use for Gaussian mixture model

    Returns
    -------
    mu : float
        Mean for 1-D Gaussian model
    std : float
        Standard deviation for 1-D Gaussian model
    """

    # * 1-D gaussian model fitting for all mutations
    x = df
    X = x.values.flatten()
    X = X[~np.isnan(X)]
    X = np.expand_dims(X, 1)

    # * build and fit Gaussian mixture model
    model = GaussianMixture(n_components=n_components, covariance_type="full")
    model.fit(X)
    means = model.means_
    covars = model.covariances_

    # * determine cutoffs significance
    # calculate standard deviation from covariances
    stdevs = np.sqrt(covars)

    # * central peak
    central_mu_idx = np.abs(means).argmin()
    mu = means[central_mu_idx]
    std = stdevs[central_mu_idx]

    return mu, std
