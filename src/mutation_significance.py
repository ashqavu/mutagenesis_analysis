#!/usr/bin/env python
"""
This script uses Gaussian modeling to determine which mutations are significant
and visualize them in scatterplots
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse, Rectangle
from sklearn.mixture import GaussianMixture

from visualization import filter_fitness_read_noise
from plasmid_map import Gene


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
    sigma_cutoff: int = 3,
    ax: matplotlib.axes = None,
    **kwargs,
) -> tuple[float, float, float]:
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
        How many sigmas to draw and final sigma for determining significance, by default 3
    ax : matplotlib.axes, optional
        Axes to draw the ellipses on, by default None

    Returns
    -------
    tuple[float, float, float]
        Width, height, and angle of final ellipse
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
    return (sigma_cutoff * width) / 2, (sigma_cutoff * height) / 2, np.radians(angle)


def gaussian_significance_model(
    x: str,
    y: str,
    counts_dict: dict,
    fitness_dict: dict,
    gene: Gene,
    draw: bool = True,
    ax: matplotlib.axes = None,
    read_threshold: int = 20,
    sigma_cutoff: int = 3,
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
         Dictionary containing all samples and DataFrames with mutation count values
    fitness_dict : dict
         Dictionary containing all samples and DataFrames with mutation fitness values
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
        cutoff for significance, by default 3
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
    X = X[~np.isnan(X).any(axis=1)]

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
        x_center = center[0]
        y_center = center[1]
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


def shish_kabob_plot(
    drug: str,
    counts_dict: dict,
    fitness_dict: dict,
    gene: Gene,
    sign_resistant_df_dict: dict,
    sign_sensitive_df_dict: dict,
    ax: matplotlib.axes,
    read_threshold: int = 20,
    orientation: str = "horizontal",
    vmin: float = -1.5,
    vmax: float = 1.5,
    cbar=False,
) -> matplotlib.axes:
    """
    Shish kabob plot showing only positions with significant mutations plotted

    Parameters
    ----------
    drug : str
        Name of drug to plot
    counts_dict : dict
        Reference with counts DataFrames for all samples
    fitness_dict : dict
        Reference with fitness DataFrames for all samples
    gene : Gene
        Gene object for locating wild-type residues
    sign_resistant_df_dict : dict
        DataFrame of bool values indicating whether the significance of the
        mutation is True or False for resistance
    sign_sensitive_df_dict : dict
        DataFrame of bool values indicating whether the significance of the
        mutation is True or False for sensitivity
    ax : matplotlib.axes
        Axes to draw the plot on
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
    # * pick pairs for each drug
    treatment_pair = [key for key in fitness_dict if drug in key]
    if not treatment_pair:
        raise KeyError(f"No fitness data: {drug}")
    if len(treatment_pair) > 2:
        raise IndexError("Treatment has more than 2 replicates to compare")

    replica_one, replica_two = treatment_pair[0], treatment_pair[1]

    sign_positions = (
        sign_resistant_df_dict[drug].drop("*", axis=1)
        | sign_sensitive_df_dict[drug].drop("*", axis=1)
    ).sum(axis=1) > 0
    sign_positions = sign_positions[sign_positions].index

    df1 = filter_fitness_read_noise(
        replica_one, counts_dict, fitness_dict, gene, read_threshold=read_threshold
    )
    df2 = filter_fitness_read_noise(
        replica_two, counts_dict, fitness_dict, gene, read_threshold=read_threshold
    )
    # find fitness value of greatest magnitude between pair
    df = df1[df1.abs().ge(df2.abs())]
    df.update(df2[df2.abs().ge(df1.abs())])
    # select only positions with significant mutations
    df_masked = df.where(sign_sensitive_df_dict[drug] | sign_resistant_df_dict[drug])
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


def drug_pair(
    drug1: str,
    drug2: str,
    sign_resistant_df_dict: dict,
    sign_sensitive_df_dict: dict,
    counts_dict: dict,
    fitness_dict: dict,
    gene: Gene,
    ax: matplotlib.axes = None,
    read_threshold: int = 20,
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
    sign_resistant_df_dict : dict
        DataFrame of bool values indicating whether the significance of the
        mutation is True or False for resistance
    sign_sensitive_df_dict : dict
        DataFrame of bool values indicating whether the significance of the
        mutation is True or False for sensitivity
    counts_dict : dict
        Reference with counts DataFrames for all samples
    fitness_dict : dict
        Reference with fitness DataFrames for all samples
    gene : Gene
        Gene object for locating wild-type residues
    ax : matplotlib.axes, optional
        Axes to draw the plot on, by default None
    read_threshold : int, optional
        Minimum number of reads required to be included, by default 20
    xlim : tuple[float, float], optional
        X-axis limits of figure, by default (-2.5, 2.5)
    ylim : tuple[float, float], optional
        y-axis limits of figure, by default (-2.5, 2.5)
    """
    if ax is None:
        ax = plt.gca()
    # * get replicate pairs for each drug
    # drug 1
    drug1_pair = [key for key in fitness_dict if drug1 in key]
    drug1_x, drug1_y = drug1_pair[0], drug1_pair[1]
    df1_x = filter_fitness_read_noise(
        drug1_x, counts_dict, fitness_dict, gene, read_threshold=read_threshold
    )
    df1_y = filter_fitness_read_noise(
        drug1_y, counts_dict, fitness_dict, gene, read_threshold=read_threshold
    )
    # drug2
    drug2_pair = [key for key in fitness_dict if drug2 in key]
    drug2_x, drug2_y = drug2_pair[0], drug2_pair[1]
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
    # drug 1 mutations
    sns.scatterplot(
        x=df1_xy[sign_resistant_df_dict[drug1]].values.flatten(),
        y=df2_xy[sign_resistant_df_dict[drug1]].values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="lightcoral",
        lw=2,
        s=10,
    )
    # drug 2 mutations
    sns.scatterplot(
        x=df1_xy[sign_resistant_df_dict[drug2]].values.flatten(),
        y=df2_xy[sign_resistant_df_dict[drug2]].values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="lightcoral",
        lw=2,
        s=10,
    )
    # drug1-drug2 shared mutations
    shared_resistant_1 = df1_xy.where(
        sign_resistant_df_dict[drug1] & sign_resistant_df_dict[drug2]
    )
    shared_resistant_2 = df2_xy.where(
        sign_resistant_df_dict[drug1] & sign_resistant_df_dict[drug2]
    )
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
    # drug 1 mutations
    sns.scatterplot(
        x=df1_xy[sign_sensitive_df_dict[drug1]].values.flatten(),
        y=df2_xy[sign_sensitive_df_dict[drug1]].values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="dodgerblue",
        lw=2,
        s=10,
    )
    # drug 2 mutations
    sns.scatterplot(
        x=df1_xy[sign_sensitive_df_dict[drug2]].values.flatten(),
        y=df2_xy[sign_sensitive_df_dict[drug2]].values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="dodgerblue",
        lw=2,
        s=10,
    )
    # drug1-drug2 shared mutations
    shared_sensitive_1 = df1_xy.where(
        sign_sensitive_df_dict[drug1] & sign_sensitive_df_dict[drug2]
    )
    shared_sensitive_2 = df2_xy.where(
        sign_sensitive_df_dict[drug1] & sign_sensitive_df_dict[drug2]
    )
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
