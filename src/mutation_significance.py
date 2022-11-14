#!/usr/bin/env python
"""
This script uses Gaussian modeling to determine which mutations are significant
and visualize them in scatterplots
"""

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse, Rectangle
from sklearn.mixture import GaussianMixture

from plasmid_map import Gene
from visualization import filter_fitness_read_noise, get_pairs


def ellipse_coordinates(covariance: np.ndarray) -> tuple[float, float, float]:
    """
    Given a position and covariance, calculate the boundaries of the ellipse to be drawn

    Parameters
    ----------
    covariance : np.ndarray
        Covariance array

    Returns
    -------
    width, height, angle : tuple[int, int, int]
        Width, height, and angle of the ellipse
    """

    # convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, _ = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    return width, height, angle


def ellipses_draw(
    center: tuple[float, float],
    width: float,
    height: float,
    angle: float,
    sigma_cutoff: int = 4,
    ax: matplotlib.axes = None,
    **kwargs,
) -> None:
    """
    Draw the ellipses for each sigma cutoff and return the parameters of the
    final ellipses

    Parameters
    ----------
    center : tuple[float, float]
        Coordinates of the center of the ellipse
    width : float
        Width of the ellipse
    height : float
        Height of the ellipse
    angle : float
        Angle of the ellipse
    sigma_cutoff : int, optional
        How many sigmas to draw and final sigma for determining significance, by default 4
    ax : matplotlib.axes, optional
        Axes to draw the ellipses on, by default None

    Returns
    -------
    None
    """
    if ax is None:
        ax = plt.gca()
    for n_sigma in range(1, sigma_cutoff + 1):
        ax.add_patch(
            Ellipse(
                center,
                n_sigma * width,
                n_sigma * height,
                angle=angle,
                ec="k",
                lw=0.667,
                fill=None,
                **kwargs,
            )
        )


def gaussian_significance_model(
    x: str,
    y: str,
    counts_dict: dict,
    fitness_dict: dict,
    gene: Gene,
    draw: bool = True,
    ax: matplotlib.axes = None,
    read_threshold: int = 20,
    sigma_cutoff: int = 4,
    xlim: tuple[float, float] = (-2.5, 2.5),
    ylim: tuple[float, float] = (-2.5, 2.5),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Determine significant mutations by using a Gaussian model to calculate
    which mutations have fitness values that are significantly different from
    synonymous mutations according to the sigma cutoff

    Parameters
    ----------
    x : str
        First sample to compare
    y : str
        Second sample to compare
    counts_dict : dict
        Reference with counts dataframes for all samples
    fitness_dict : dict
        Reference with fitness dataframes for all samples
    gene : Gene
        Gene object for locating wild-type residues
    draw : bool, optional
        Whether to draw the ellipses or just return boundaries, by default True
    ax : matplotlib.axes, optional
        Axes to draw the ellipses on, by default None
    read_threshold : int, optional
        Minimum number of reads required to be included, by default 20
    sigma_cutoff : int, optional
        How many sigmas away from the synonymous mutation values to use as the
        cutoff for significance, by default 4
    xlim : tuple[float, float], optional
        X-axis limits of figure, by default (-2.5, 2.5)
    ylim : tuple[float, float], optional
        Y-axis limits of figure, by default (-2.5, 2.5)

    Returns
    -------
    sign_resistant, sign_sensitive : tuple[pd.DataFrame, pd.DataFrame]
        DataFrame of bool values indicating whether the significance of the
        mutation is True or False for resistance or sensitivity
    """

    if ax is None and draw:
        ax = plt.gca()
    df_x = filter_fitness_read_noise(x, counts_dict, fitness_dict, gene, read_threshold)
    df_y = filter_fitness_read_noise(y, counts_dict, fitness_dict, gene, read_threshold)

    # * merge two dataframes element-wise for significance test
    df_xy = pd.concat([df_x, df_y]).groupby(level=0, axis=0).agg(list)

    # * select synonymous mutations
    x_syn = df_x["∅"]
    y_syn = df_y["∅"]
    # * build numpy matrix for gaussian model fitting
    X_syn_train = np.column_stack((x_syn.values.flatten(), y_syn.values.flatten()))
    # filter NaN in pairs
    X_syn_train = X_syn_train[~np.isnan(X_syn_train).any(axis=1)]

    # * train model off of synonymous mutation data
    gaussian_model = GaussianMixture(
        n_components=1, covariance_type="full", means_init=[[0, 0]]
    )
    gaussian_model.fit(X_syn_train)

    # * build numpy matrix for gaussian model plotting
    X = np.column_stack((df_x.values.flatten(), df_y.values.flatten()))
    # filter NaN in pairs
    X = X[np.isfinite(X).all(axis=1)]

    # * drawing ellipses given position and covariance
    for center, covar in zip(gaussian_model.means_, gaussian_model.covariances_):
        # calculate ellipse paramaters
        width, height, angle = ellipse_coordinates(covar)
        if draw:
            # draw ellipses
            ellipses_draw(
                center,
                width,
                height,
                angle,
                sigma_cutoff=sigma_cutoff,
                ax=ax,
                zorder=10,
            )
        width_f, height_f, angle_f = (
            (sigma_cutoff * width) / 2,
            (sigma_cutoff * height) / 2,
            np.radians(angle),
        )
    # * determine significance
    def is_outside_ellipse(point):
        # if either fitness value is NaN, discard point
        if np.isnan(point).any():
            return False
        x, y = point
        cos_angle = np.cos(angle_f)
        sin_angle = np.sin(angle_f)
        x_center, y_center = center[0], center[1]
        x_dist = x - x_center
        y_dist = y - y_center
        a = ((x_dist * cos_angle) + (y_dist * sin_angle)) ** 2
        b = ((x_dist * sin_angle) - (y_dist * cos_angle)) ** 2
        limit = (width_f**2) * (height_f**2)
        return a * (height_f**2) + b * (width_f**2) > limit

    def is_positive_quadrant(point):
        return (np.array(point) >= 0).all()

    def is_negative_quadrant(point):
        return (np.array(point) <= 0).all()

    def is_sign_resistant(point):
        return is_outside_ellipse(point) and is_positive_quadrant(point)

    def is_sign_sensitive(point):
        return is_outside_ellipse(point) and is_negative_quadrant(point)

    sign_resistant = df_xy.applymap(is_sign_resistant)
    sign_sensitive = df_xy.applymap(is_sign_sensitive)

    if draw:
        # * plotting
        # all mutations
        sns.scatterplot(
            x=X[:, 0],
            y=X[:, 1],
            zorder=-1,
            ax=ax,
            plotnonfinite=False,
            color="gray",
            lw=2,
            s=10,
        )
        # synonymous mutations
        sns.scatterplot(
            x=df_x["∅"],
            y=df_y["∅"],
            ax=ax,
            plotnonfinite=False,
            color="yellowgreen",
            lw=0.5,
            s=10,
        )
        # resistant mutations
        sns.scatterplot(
            x=df_x[sign_resistant].values.flatten(),
            y=df_y[sign_resistant].values.flatten(),
            ax=ax,
            plotnonfinite=False,
            color="lightcoral",
            lw=0.5,
            s=10,
        )
        # sensitive mutations
        sns.scatterplot(
            x=df_x[sign_sensitive].values.flatten(),
            y=df_y[sign_sensitive].values.flatten(),
            ax=ax,
            plotnonfinite=False,
            color="dodgerblue",
            lw=0.5,
            s=10,
        )

        # * axis lines and limits
        # diagonal
        ax.plot([-4, 4], [-4, 4], ":", color="gray", alpha=0.5, zorder=0)
        # x-axis
        ax.plot([0, 0], [-4, 4], "-", color="gray", alpha=0.5, lw=1, zorder=0)
        # y y-axis
        ax.plot([-4, 4], [0, 0], "-", color="gray", alpha=0.5, lw=1, zorder=0)
        ax.set(xlim=xlim, ylim=ylim, anchor="NW", aspect="equal")
        ax.set_xlabel(x, fontweight="bold")
        ax.set_ylabel(y, fontweight="bold")

    return sign_resistant, sign_sensitive


def gaussian_replica_pair_draw(
    counts_dict: dict,
    fitness_dict: dict,
    gene: Gene,
    read_threshold: int = 20,
    sigma_cutoff: int = 4,
    xlim: tuple[float, float] = (-2.5, 2.5),
    ylim: tuple[float, float] = (-2.5, 2.5),
) -> matplotlib.figure:
    """
    Draws the full figure of gaussian significance scatterplots for all drugs
    in experiment. All treated-untreated pairs must be present in the
    dictionary.

    Parameters
    ----------
    counts_dict : dict
        Reference with counts dataframes for all samples
    fitness_dict : dict
        Reference with fitness dataframes for all samples
    gene : Gene
        Gene object for locating wild-type residues
    read_threshold : int, optional
        Minimum number of reads required to be included, by default 20
    sigma_cutoff : int, optional
        How many sigmas away from the synonymous mutation values to use as the
        cutoff for significance, by default 4
    xlim : tuple[float, float], optional
        X-axis limits of figure, by default (-2.5, 2.5)
    ylim : tuple[float, float], optional
        Y-axis limits of figure, by default (-2.5, 2.5)

    Returns
    -------
    matplotlib.figure
    """
    # * determine shape of subplots
    drugs_all = sorted(set(x.rstrip("1234567890") for x in fitness_dict))
    num_plots = len(drugs_all)
    rows = cols = np.sqrt(num_plots)
    if not rows.is_integer():
        rows, cols = np.floor(rows), np.ceil(cols)
        if num_plots > rows * cols:
            rows += 1
    rows = int(rows)
    cols = int(cols)

    # * begin drawing
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10), layout="compressed")
    sign_resistant_df_bools = {}
    sign_sensitive_df_bools = {}
    for i, drug in enumerate(sorted(drugs_all)):
        # * pick pairs for each drug
        replica_one, replica_two = get_pairs(drug, fitness_dict)

        ax = axs.flat[i]
        sign_resistant, sign_sensitive = gaussian_significance_model(
            replica_one,
            replica_two,
            counts_dict=counts_dict,
            fitness_dict=fitness_dict,
            gene=gene,
            draw=True,
            read_threshold=read_threshold,
            sigma_cutoff=sigma_cutoff,
            ax=ax,
            xlim=xlim,
            ylim=ylim,
        )
        sign_resistant_df_bools[drug] = sign_resistant
        sign_sensitive_df_bools[drug] = sign_sensitive
    while len(fig.axes) > num_plots:
        fig.axes[-1].remove()
    fig.get_layout_engine().set(hspace=0.1, wspace=0.1)
    return fig


def significant_sigma_bools(
    counts_dict: dict,
    fitness_dict: dict,
    gene: Gene,
    read_threshold: int = 20,
    sigma_cutoff: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts the significant residue positions from the fitness dataframes

    Parameters
    ----------
    counts_dict : dict
        Reference for counts values of all samples
    fitness_dict : dict
        Reference for fitness values of all samples
    gene : Gene
        Gene object for locating wild-type residues
    read_threshold : int, optional
        Minimum number of reads required to be included, by default 20
    sigma_cutoff : int, optional
        How many sigmas away from the synonymous mutation values to use as the
        cutoff for significance, by default 4

    Returns
    -------
    sign_resistant_df_bools, sign_sensitive_df_bools : tuple[pd.DataFrame, pd.DataFrame]
        Dataframes of boolean values indicating which cells of the table are
        relevant mutations for the drug
    """
    sign_resistant_df_bools = {}
    sign_sensitive_df_bools = {}
    drugs = set([x.rstrip("1234567890") for x in fitness_dict])
    for drug in drugs:
        replica_one, replica_two = get_pairs(drug, fitness_dict)
        sign_resistant, sign_sensitive = gaussian_significance_model(
            replica_one,
            replica_two,
            counts_dict=counts_dict,
            fitness_dict=fitness_dict,
            gene=gene,
            read_threshold=read_threshold,
            sigma_cutoff=sigma_cutoff,
            draw=False,
        )
        sign_resistant_df_bools[drug] = sign_resistant
        sign_sensitive_df_bools[drug] = sign_sensitive
    return sign_resistant_df_bools, sign_sensitive_df_bools


def shish_kabob_plot(
    drug: str,
    counts_dict: dict,
    fitness_dict: dict,
    gene: Gene,
    sign_resistant: pd.DataFrame,
    sign_sensitive: pd.DataFrame,
    ax: matplotlib.axes = None,
    read_threshold: int = 20,
    orientation: str = "horizontal",
    vmin: float = -1.5,
    vmax: float = 1.5,
    cbar: bool = False,
) -> matplotlib.axes:
    """
    Shish kabob plot showing only positions with significant mutations plotted

    Parameters
    ----------
    drug : str
        Name of drug to plot
    counts_dict : dict
        Reference with counts dataframes for all samples
    fitness_dict : dict
        Reference with fitness dataframes for all samples
    gene : Gene
        Gene object for locating wild-type residues
    sign_resistant : pd.DataFrame
        DataFrame of bool values indicating whether the significance of the
        mutation is True or False for resistance
    sign_sensitive : pd.DataFrame
        DataFrame of bool values indicating whether the significance of the
        mutation is True or False for sensitivity
    ax : matplotlib.axes, optional
        Axes to draw the plot on, by default None
    read_threshold : int, optional
        Minimum number of reads required to be included, by default 20
    orientation : str, optional
        Whether to draw plot vertically or horizontally, by default "horizontal"
    vmin : float, optional
        For fitness data, vmin parameter passed to sns.heatmap, by default -1.5
    vmax : float, optional
        For fitness data, vmax parameter passed to sns.heatmap, by default 1.5
    cbar : bool, optional
        Whether to draw colorbar or not, by default False

    Returns
    -------
    ax : matplotlib.axes

    Raises
    ------
    KeyError
        No fitness data found for the specified drug
    IndexError
        Will run into problems with graphing if there are more than two
        replicates in the group
    """
    if ax is None:
        ax = plt.gca()
    # * get residue positions with significant mutation
    sign_positions = (
        sign_resistant.drop("*", axis=1) | sign_sensitive.drop("*", axis=1)
    ).sum(axis=1) > 0
    sign_positions = sign_positions[sign_positions].index

    # * pick pairs for each drug
    replica_one, replica_two = get_pairs(drug, fitness_dict)

    # * make sure the fitness values are filtered for counts above the read threshold
    df1 = filter_fitness_read_noise(
        replica_one, counts_dict, fitness_dict, gene, read_threshold=read_threshold
    )
    df2 = filter_fitness_read_noise(
        replica_two, counts_dict, fitness_dict, gene, read_threshold=read_threshold
    )
    # * find fitness value of greatest magnitude between pair
    df = df1[df1.abs().ge(df2.abs())]
    df.update(df2[df2.abs().ge(df1.abs())])
    # * select only positions with significant mutations
    df_masked = df.where(sign_resistant | sign_sensitive)
    df_masked = df_masked.drop("∅", axis=1)

    with sns.axes_style("white"):
        if orientation == "vertical":
            df_masked_plot = df_masked.loc[sign_positions]

            sns.heatmap(
                df_masked_plot,
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                cbar=cbar,
                square=True,
                ax=ax,
            )
            ax.yaxis.grid("on")
            ax.set_yticks(
                np.arange(len(sign_positions)) + 0.5,
                np.array(sign_positions),
                rotation=0,
                ha="center",
                fontsize="xx-small",
            )
            ax.set_xticks([])
            # * add wild-type notations
            # get reference residues
            ref_aas = np.take(gene.cds_translation, sign_positions)
            # iterate over amino acid options (y-axis)
            for y, residue in enumerate(ref_aas):
                # determine x position for text box
                x = df_masked_plot.columns.get_loc(residue)
                ax.add_patch(
                    Rectangle((x, y), 1, 1, ec="black", fc="white", fill=True, lw=0.2)
                )
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    residue,
                    fontsize="x-small",
                    ha="center",
                    va="center",
                )
            # * annotate fitness boxes
            # iterate over the x-axis (positions)
            for x, aa in enumerate(df_masked_plot.columns):
                # the significant positions of each amino acid mutation
                # determinates y-coord for text box
                pos_indices = np.argwhere(df_masked_plot[aa].notnull().values)
                for y in pos_indices:
                    ax.text(
                        x + 0.5,
                        y + 0.5,
                        aa,
                        fontsize="x-small",
                        ha="center",
                        va="center",
                        color="white",
                    )
            ax.set_title(drug, fontweight="heavy")
            ax.set_anchor("N")

        elif orientation == "horizontal":
            df_masked = df_masked.T
            df_masked_plot = df_masked[sign_positions]

            sns.heatmap(
                df_masked_plot,
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                cbar=cbar,
                square=True,
                ax=ax,
            )
            ax.xaxis.grid("on")
            ax.set_xticks(
                np.arange(len(sign_positions)) + 0.5,
                np.array(sign_positions),
                rotation=90,
                ha="center",
                fontsize="xx-small",
            )
            ax.set_yticks([])
            # * annotate fitness boxes
            # get reference residues
            ref_aas = np.take(gene.cds_translation, sign_positions)
            # iterate in the x-direction (significant positions)
            for x, residue in enumerate(ref_aas):
                # determine y-coord for text box
                y = df_masked.index.get_loc(residue)
                ax.add_patch(
                    Rectangle(
                        (x, y),
                        1,
                        1,
                        ec="black",
                        fc="white",
                        fill=True,
                        lw=0.2,
                        clip_on=False,
                    )
                )
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    residue,
                    fontsize="x-small",
                    ha="center",
                    va="center",
                )
            for x, pos in enumerate(sign_positions):
                aa_indices = np.argwhere(df_masked_plot[pos].notnull().values)
                for y in aa_indices:
                    aa = df_masked_plot.index[y].values[0]
                    ax.text(
                        x + 0.5,
                        y + 0.5,
                        aa,
                        fontsize="x-small",
                        ha="center",
                        va="center",
                        color="white",
                    )
            ax.set_ylabel(drug, fontweight="heavy")
            ax.set_anchor("W")
        return ax


def shish_kabob_plot_draw(
    counts_dict: dict,
    fitness_dict: dict,
    gene: Gene,
    read_threshold: int = 20,
    sigma_cutoff: int = 4,
    orientation: str = "horizontal",
    vmin: float = -1.5,
    vmax: float = 1.5,
    xlim: tuple[float, float] = (-2.5, 2.5),
    ylim: tuple[float, float] = (-2.5, 2.5),
) -> matplotlib.axes:
    """
    Draw shish kabob plots and corresponding gaussian scatter plots for all
    samples in dataset

    Parameters
    ----------
    counts_dict : dict
        Reference with counts dataframes for all samples
    fitness_dict : dict
        Reference with fitness dataframes for all samples
    gene : Gene
        Gene object for locating wild-type residues
    read_threshold : int, optional
        Minimum number of reads required to be included, by default 20
    sigma_cutoff : int, optional
        How many sigmas away from the synonymous mutation values to use as the
        cutoff for significance, by default 4
    orientation : str, optional
        Whether to draw plot vertically or horizontally, by default "horizontal"
    vmin : float, optional
        For fitness data, vmin parameter passed to sns.heatmap, by default -1.5
    vmax : float, optional
        For fitness data, vmax parameter passed to sns.heatmap, by default 1.5
    xlim : tuple[float, float], optional
        X-axis limits of gaussian figure, by default (-2.5, 2.5)
    ylim : tuple[float, float], optional
        Y-axis limits of gaussian figure, by default (-2.5, 2.5)
    """
    # * determine shape of subplots
    drugs_all = sorted(set(x.rstrip("1234567890") for x in fitness_dict))
    gridspec_dict = {"wspace": 0, "hspace": 0}
    if orientation == "horizontal":
        num_rows, num_cols = len(drugs_all), 2
        gridspec_dict.update({"width_ratios": [2.5, 1]})
        figsize = (7, 17)
    elif orientation == "vertical":
        num_rows, num_cols = 2, len(drugs_all)
        gridspec_dict.update({"height_ratios": [2.5, 1]})
        figsize = (17, 7)

    with sns.axes_style("white"):
        fig, axs = plt.subplots(
            num_rows,
            num_cols,
            figsize=figsize,
            dpi=300,
            layout="constrained",
            gridspec_kw=gridspec_dict,
        )
        fig.suptitle(
            f"Significant mutations (min. read = {read_threshold})", fontweight="heavy"
        )
        for i, drug in enumerate(drugs_all):
            # * determine subplots for shish kabob and gaussians
            if orientation == "horizontal":
                ax_shish = axs[i, 0]
                ax_gauss = axs[i, 1]
            elif orientation == "vertical":
                ax_shish = axs[0, i]
                ax_gauss = axs[1, i]
            ax_gauss.set_xlabel(f"{drug}1", size="x-small")
            ax_gauss.set_ylabel(f"{drug}2", size="x-small")
            ax_gauss.tick_params(labelsize="xx-small")
            ax_gauss.set_anchor("W")

            replica_one, replica_two = get_pairs(drug, fitness_dict)
            sign_resistant, sign_sensitive = gaussian_significance_model(
                replica_one,
                replica_two,
                counts_dict,
                fitness_dict,
                gene,
                ax=ax_gauss,
                read_threshold=read_threshold,
                sigma_cutoff=sigma_cutoff,
                xlim=xlim,
                ylim=ylim,
            )

            shish_kabob_plot(
                drug,
                counts_dict,
                fitness_dict,
                gene,
                sign_resistant,
                sign_sensitive,
                ax=ax_shish,
                read_threshold=read_threshold,
                orientation=orientation,
                vmin=vmin,
                vmax=vmax,
            )

    return fig


def drug_pair(
    drug1: str,
    drug2: str,
    counts_dict: dict,
    fitness_dict: dict,
    gene: Gene,
    ax: matplotlib.axes = None,
    read_threshold: int = 20,
    sigma_cutoff: int = 4,
    xlim: tuple[float, float] = (-2.5, 2.5),
    ylim: tuple[float, float] = (-2.5, 2.5),
) -> None:
    """
    Find common significant resistance/sensitivity mutations between two different drugs

    Parameters
    ----------
    drug1 : str
        First drug
    drug2 : str
        Second drug
    sign_resistant_df_bools : dict
        dataframes of bool values indicating whether the significance of the
        mutation is True or False for resistance
    sign_sensitive_df_bools : dict
        DataFrame of bool values indicating whether the significance of the
        mutation is True or False for sensitivity
    counts_dict : dict
        Reference with counts dataframes for all samples
    fitness_dict : dict
        Reference with fitness dataframes for all samples
    gene : Gene
        Gene object for locating wild-type residues
    ax : matplotlib.axes, optional
        Axes to draw the plot on, by default
    read_threshold : int, optional
        Minimum number of reads required to be included, by default 20
    sigma_cutoff : int, optional
        How many sigmas away from the synonymous mutation values to use as the
        cutoff for significance, by default 4
    xlim : tuple[float, float], optional
        X-axis limits of figure, by default (-2.5, 2.5)
    ylim : tuple[float, float], optional
        y-axis limits of figure, by default (-2.5, 2.5)
    """
    if ax is None:
        ax = plt.gca()
    # * get cells of significant mutations
    # drug 1
    drug1_x, drug1_y = get_pairs(drug1, fitness_dict)
    sign_resistant_df_bool1, sign_sensitive_df_bool1 = gaussian_significance_model(
        drug1_x,
        drug1_y,
        counts_dict,
        fitness_dict,
        gene,
        draw=False,
        read_threshold=read_threshold,
        sigma_cutoff=sigma_cutoff,
    )
    # drug 2
    drug2_x, drug2_y = get_pairs(drug2, fitness_dict)
    sign_resistant_df_bool2, sign_sensitive_df_bool2 = gaussian_significance_model(
        drug2_x,
        drug2_y,
        counts_dict,
        fitness_dict,
        gene,
        draw=False,
        read_threshold=read_threshold,
        sigma_cutoff=sigma_cutoff,
    )

    # * get filtered fitnesses
    # drug 1
    df1_x = filter_fitness_read_noise(
        drug1_x, counts_dict, fitness_dict, gene, read_threshold=read_threshold
    )
    df1_y = filter_fitness_read_noise(
        drug1_y, counts_dict, fitness_dict, gene, read_threshold=read_threshold
    )
    # drug2
    df2_x = filter_fitness_read_noise(
        drug2_x, counts_dict, fitness_dict, gene, read_threshold=read_threshold
    )
    df2_y = filter_fitness_read_noise(
        drug2_y, counts_dict, fitness_dict, gene, read_threshold=read_threshold
    )
    # * find mean of fitness values between replicates
    df1_xy = pd.concat([df1_x, df1_y]).groupby(level=0, axis=0).agg(np.mean)
    df2_xy = pd.concat([df2_x, df2_y]).groupby(level=0, axis=0).agg(np.mean)
    # * build numpy matrix of all points for plotting
    X = np.column_stack((df1_xy.values.flatten(), df2_xy.values.flatten()))
    X = X[np.isfinite(X).all(axis=1)]

    # * scatterplots
    # all mutations
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        zorder=-1,
        ax=ax,
        plotnonfinite=False,
        color="gray",
        lw=2,
        s=10,
    )
    # * resistance mutations
    # drug 1 resistance mutations
    sns.scatterplot(
        x=df1_xy[sign_resistant_df_bool1].values.flatten(),
        y=df2_xy[sign_resistant_df_bool1].values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="lightcoral",
        lw=2,
        s=10,
    )
    # drug 2 resistance mutations
    sns.scatterplot(
        x=df1_xy[sign_resistant_df_bool2].values.flatten(),
        y=df2_xy[sign_resistant_df_bool2].values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="lightcoral",
        lw=2,
        s=10,
    )
    # drug1-drug2 shared resistance mutations
    shared_resistant_1 = df1_xy.where(sign_resistant_df_bool1 & sign_resistant_df_bool2)
    shared_resistant_2 = df2_xy.where(sign_resistant_df_bool1 & sign_resistant_df_bool2)
    sns.scatterplot(
        x=shared_resistant_1.values.flatten(),
        y=shared_resistant_2.values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="firebrick",
        lw=2,
        s=10,
        marker="D",
    )
    # * sensitive mutations
    # drug 1 sensitive mutations
    sns.scatterplot(
        x=df1_xy[sign_sensitive_df_bool1].values.flatten(),
        y=df2_xy[sign_sensitive_df_bool1].values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="dodgerblue",
        lw=2,
        s=10,
    )
    # drug 2 sensitive mutations
    sns.scatterplot(
        x=df1_xy[sign_sensitive_df_bool2].values.flatten(),
        y=df2_xy[sign_sensitive_df_bool2].values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="dodgerblue",
        lw=2,
        s=10,
    )
    # drug1-drug2 shared sensitive mutations
    shared_sensitive_1 = df1_xy.where(sign_sensitive_df_bool1 & sign_sensitive_df_bool2)
    shared_sensitive_2 = df2_xy.where(sign_sensitive_df_bool1 & sign_sensitive_df_bool2)
    sns.scatterplot(
        x=shared_sensitive_1.values.flatten(),
        y=shared_sensitive_2.values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="mediumblue",
        lw=2,
        s=10,
        marker="D",
    )

    ax.plot([-4, 4], [-4, 4], ":", color="gray", alpha=0.5, zorder=0)
    ax.plot([0, 0], [-4, 4], "-", color="gray", alpha=0.5, lw=1, zorder=0)
    ax.plot([-4, 4], [0, 0], "-", color="gray", alpha=0.5, lw=1, zorder=0)
    ax.set(xlim=xlim, ylim=ylim)
    ax.tick_params(left=False, bottom=False, labelsize="xx-small")
    ax.set_anchor("NW")
    ax.set_aspect("equal")


def significant_sigma_mutations(
    counts_dict: dict,
    fitness_dict: dict,
    gene,
    read_threshold: int = 20,
    sigma_cutoff: int = 4,
) -> pd.DataFrame:
    cds_translation = gene.cds_translation
    drugs = sorted(set([x.rstrip("1234567890") for x in fitness_dict]))
    sign_resistant_df_dict, sign_sensitive_df_dict = significant_sigma_bools(
        counts_dict,
        fitness_dict,
        gene,
        read_threshold=read_threshold,
        sigma_cutoff=sigma_cutoff,
    )

    sample_name = list(counts_dict)[0]
    point_mutations = pd.MultiIndex.from_product(
        [fitness_dict[sample_name].index, fitness_dict[sample_name].columns]
    ).values
    list_all_fitness = []
    for drug in drugs:
        replica_one, replica_two = get_pairs(drug, fitness_dict)
        df1 = fitness_dict[replica_one]
        df2 = fitness_dict[replica_two]

        sign_resistant = sign_resistant_df_dict[drug]
        sign_sensitive = sign_sensitive_df_dict[drug]

        for position, residue in point_mutations:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                fitness_entry = {
                    "aa_pos": position,
                    "ref_aa": cds_translation[position],
                    "query_aa": residue,
                    "drug": drug,
                    "rel_fitness_1": df1.loc[position, residue],
                    "rel_fitness_2": df2.loc[position, residue],
                    "rel_fitness_mean": np.nanmean(
                        [df1.loc[position, residue], df2.loc[position, residue]]
                    ),
                    "significant": (
                        sign_sensitive.loc[position, residue]
                        | sign_resistant.loc[position, residue]
                    ),
                }
            if fitness_entry["significant"]:
                if sign_sensitive.loc[position, residue]:
                    fitness_entry.update({"type": "sensitive"})
                elif sign_resistant.loc[position, residue]:
                    fitness_entry.update({"type": "resistance"})
            list_all_fitness.append(fitness_entry)

    df_all_fitness_sigma = pd.DataFrame(list_all_fitness)
    return df_all_fitness_sigma
