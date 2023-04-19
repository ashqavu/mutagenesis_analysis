#!/usr/bin/env python
"""
Determine and plot Gaussian significance models for drugs used in mutagenesis studies
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Ellipse

from fitness_analysis import gaussian_significance_2d
from sequencing_data import SequencingData
from utils.seq_data_utils import heatmap_masks


def gaussian_drug_2d(
    drug: str,
    data: SequencingData,
    read_threshold: int = 20,
    sigma_cutoff: int = 4,
    ax: matplotlib.axes = None,
    xlim: tuple[float, float] = (-2.5, 2.5),
    ylim: tuple[float, float] = (-2.5, 2.5),
) -> matplotlib.axes:
    counts_dict = data.counts
    fitness_dict = data.fitness
    gene = data.gene
    wt_mask = heatmap_masks(gene)

    x, y = data.get_pairs(drug, data.samples)
    dfs_filtered = data.filter_fitness_read_noise(
        counts_dict, fitness_dict, read_threshold=read_threshold
    )
    df_x = dfs_filtered[x]
    df_y = dfs_filtered[y]
    df_x = df_x.mask(wt_mask)
    df_y = df_y.mask(wt_mask)
    df_x = df_x.loc[23:285]
    df_y = df_y.loc[23:285]

    sign_sensitive, sign_resistant, ellipses_all = gaussian_significance_2d(
        df_x,
        df_y,
        sigma_cutoff=sigma_cutoff,
    )
    if ax is None:
        ax = plt.gca()

    # * draw the ellipses for each sigma cutoff
    for _, ellipse_sigma in ellipses_all.items():
        center, width, height, angle = ellipse_sigma
        ax.add_patch(
            Ellipse(
                center,
                width,
                height,
                angle=angle,
                ec="k",
                lw=0.667,
                fill=None,
                zorder=10,
            )
        )

    # * construct numpy matrix of all fitness values for plotting
    X = np.column_stack((df_x.values.flatten(), df_y.values.flatten()))
    # filter NaN in pairs
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
    # synonymous mutations
    sns.scatterplot(
        x=df_x["∅"],
        y=df_y["∅"],
        ax=ax,
        plotnonfinite=False,
        color="yellowgreen",
        lw=0.5,
        s=5,
    )
    # resistant mutations
    sns.scatterplot(
        x=df_x[sign_resistant].values.flatten(),
        y=df_y[sign_resistant].values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="lightcoral",
        lw=0.5,
        s=5,
    )
    # sensitive mutations
    sns.scatterplot(
        x=df_x[sign_sensitive].values.flatten(),
        y=df_y[sign_sensitive].values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="dodgerblue",
        lw=0.5,
        s=5,
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

    return ax


def gaussian_replica_pair_draw_2d(
    data: SequencingData,
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
    data : SequencingData
        Data from experiment sequencing with count-, enrichment-, and fitness-values
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
    drugs_all = sorted([drug for drug in data.treatments if "UT" not in drug])
    num_plots = len(drugs_all)
    rows = cols = np.sqrt(num_plots)
    if not rows.is_integer():
        rows, cols = np.floor(rows), np.ceil(cols)
        if num_plots > rows * cols:
            rows += 1
    rows = int(rows)
    cols = int(cols)

    # * begin drawing
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10), layout="compressed", dpi=300)
    for i, drug in enumerate(sorted(drugs_all)):
        ax = axs.flat[i]
        gaussian_drug_2d(
            drug,
            data,
            read_threshold=read_threshold,
            sigma_cutoff=sigma_cutoff,
            ax=ax,
            xlim=xlim,
            ylim=ylim,
        )
    while len(fig.axes) > num_plots:
        fig.axes[-1].remove()
    fig.get_layout_engine().set(hspace=0.1, wspace=0.1)
    fig.suptitle(
        f"Significant mutations\nmin. reads = {read_threshold}, sigma cutoff = {sigma_cutoff}",
        fontweight="heavy",
        fontsize="x-large",
    )
    return fig
