#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fitness_analysis import gaussian_significance
from sequencing_data import SequencingData
from utils import (
    filter_fitness_read_noise,
    get_pairs,
    heatmap_masks,
)


def drug_pair(
    drug1: str,
    drug2: str,
    data: SequencingData,
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
    counts_dict = data.counts
    fitness_dict = data.fitness
    gene = data.gene

    if ax is None:
        ax = plt.gca()
    wt_mask = heatmap_masks(gene)
    # * get cells of significant mutations
    dfs_filtered = filter_fitness_read_noise(
        counts_dict, fitness_dict, read_threshold=read_threshold
    )
    # drug 1
    drug1_x, drug1_y = get_pairs(drug1, data.samples)
    df1_x = dfs_filtered[drug1_x].mask(wt_mask)
    df1_y = dfs_filtered[drug1_y].mask(wt_mask)
    df_sign_sensitive1, df_sign_resistant1, _ = gaussian_significance(
        df1_x, df1_y, sigma_cutoff=sigma_cutoff
    )
    # drug 2
    drug2_x, drug2_y = get_pairs(drug2, data.samples)
    df2_x = dfs_filtered[drug2_x].mask(wt_mask)
    df2_y = dfs_filtered[drug2_y].mask(wt_mask)
    df_sign_sensitive2, df_sign_resistant2, _ = gaussian_significance(
        df2_x, df2_y, sigma_cutoff=sigma_cutoff
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
        s=5,
    )
    # * sensitive mutations
    # drug 1 sensitive mutations
    sns.scatterplot(
        x=df1_xy[df_sign_sensitive1].values.flatten(),
        y=df2_xy[df_sign_sensitive1].values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="dodgerblue",
        lw=2,
        s=5,
    )
    # drug 2 sensitive mutations
    sns.scatterplot(
        x=df1_xy[df_sign_sensitive2].values.flatten(),
        y=df2_xy[df_sign_sensitive2].values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="dodgerblue",
        lw=2,
        s=5,
    )
    # drug1-drug2 shared sensitive mutations
    shared_sensitive_1 = df1_xy.where(df_sign_sensitive1 & df_sign_sensitive2)
    shared_sensitive_2 = df2_xy.where(df_sign_sensitive1 & df_sign_sensitive2)
    sns.scatterplot(
        x=shared_sensitive_1.values.flatten(),
        y=shared_sensitive_2.values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="mediumblue",
        lw=2,
        s=5,
        marker="D",
    )
    # * resistance mutations
    # drug 1 resistance mutations
    sns.scatterplot(
        x=df1_xy[df_sign_resistant1].values.flatten(),
        y=df2_xy[df_sign_resistant1].values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="lightcoral",
        lw=2,
        s=5,
    )
    # drug 2 resistance mutations
    sns.scatterplot(
        x=df1_xy[df_sign_resistant2].values.flatten(),
        y=df2_xy[df_sign_resistant2].values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="lightcoral",
        lw=2,
        s=5,
    )
    # drug1-drug2 shared resistance mutations
    shared_resistant_1 = df1_xy.where(df_sign_resistant1 & df_sign_resistant2)
    shared_resistant_2 = df2_xy.where(df_sign_resistant1 & df_sign_resistant2)
    sns.scatterplot(
        x=shared_resistant_1.values.flatten(),
        y=shared_resistant_2.values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="firebrick",
        lw=2,
        s=5,
        marker="D",
    )

    ax.plot([-4, 4], [-4, 4], ":", color="gray", alpha=0.5, zorder=0)
    ax.plot([0, 0], [-4, 4], "-", color="gray", alpha=0.5, lw=1, zorder=0)
    ax.plot([-4, 4], [0, 0], "-", color="gray", alpha=0.5, lw=1, zorder=0)
    ax.set(xlim=xlim, ylim=ylim)
    ax.tick_params(left=False, bottom=False, labelsize="xx-small")
    ax.set_anchor("NW")
    ax.set_aspect("equal")
    return ax


def drug_pairs_draw(
    data: SequencingData,
    read_threshold: int = 20,
    sigma_cutoff: int = 4,
):
    drugs_all = sorted([drug for drug in data.treatments if "UT" not in drug])
    rows = cols = len(drugs_all) - 1
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(15, 15),
        layout="compressed",
        sharex="col",
        sharey="row",
        gridspec_kw={"hspace": 0, "wspace": 0},
        dpi=300,
    )
    for ax_row, drug_y in enumerate(drugs_all):
        for ax_col, drug_x in enumerate(drugs_all):
            if ax_row > ax_col:
                ax = axs[ax_row - 1, ax_col]
                drug_pair(
                    drug_x,
                    drug_y,
                    data,
                    ax=ax,
                    read_threshold=read_threshold,
                    sigma_cutoff=sigma_cutoff,
                )
                if ax_col == 0:
                    ax.set_ylabel(drug_y)
                if ax_row == len(drugs_all) - 1:
                    ax.set_xlabel(drug_x)
            elif ax_row > 0 and (ax_col < len(drugs_all) - 1):
                ax = axs[ax_row - 1, ax_col]
                ax.set_in_layout(False)
                ax.remove()
    fig.suptitle(
        f"min. reads: {read_threshold}\nsigma cutoff: {sigma_cutoff}",
        x=0.7,
        y=0.7,
        fontsize="xx-large",
        fontweight="heavy",
    )
    return fig
