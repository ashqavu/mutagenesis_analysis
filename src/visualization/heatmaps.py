#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from plasmid_map import Gene
from sequencing_data import SequencingData
from utils import filter_fitness_read_noise, heatmap_masks
from visualization.visualization_utils import respine


def heatmap_wrapper(
    df: pd.DataFrame,
    name: str,
    dataset: str,
    gene: Gene,
    ax: matplotlib.axes = None,
    cbar: bool = False,
    cbar_ax: matplotlib.axes = None,
    vmin: float = -2.0,
    vmax: float = 2.0,
    fitness_cmap: str = "vlag",
    orientation: str = "horizontal",
) -> matplotlib.axes:
    """
    Function wrapper for preferred heatmap aesthetic settings

    Parameters
    ----------
    df : pd.DataFrame
        Matrix to be plotted
    name : str
        Sample name for axes labeling
    dataset : str
        Type of data ("counts" or "fitness")
    gene : Gene
        Gene object to provide residue numbering
    ax : matplotlib.axes, optional
        Axes on which to draw the data, by default None
    cbar : bool, optional
        Whether to draw a colorbar or not, by default False
    cbar_ax : matplotlib.axes, optional
        Axes on which to draw the colorbar, by default None
    vmin : float, optional
        For fitness data, vmin parameter passed to sns.heatmap, by default -2.0
    vmax : float, optional
        For fitness data, vmax parameter passed to sns.heatmap, by default 2.0
    fitness_cmap : str, optional
        Colormap to use for fitness heatmap, by default "vlag"
    orientation : str, optional
        Whether to draw heatmaps vertically or horizontally, by default "horizontal"

    Returns
    -------
    h : matplotlib.axes
        Axes object with the heatmap
    """
    if ax is None:
        ax = plt.gca()
    if dataset == "counts":
        with np.errstate(divide="ignore"):
            df = df.where(df.lt(1), np.log10(df))
            cmap = "Blues"
            vmin = None
            vmax = None
    elif dataset == "fitness":
        cmap = fitness_cmap

    xticklabels, yticklabels = 1, 1
    df_wt = heatmap_masks(gene)
    cbar_location = "right"

    if orientation == "horizontal":
        df_wt = df_wt.T
        df = df.T
        xticklabels, yticklabels = yticklabels, xticklabels
        cbar_location = "bottom"

    h = sns.heatmap(
        df,
        square=True,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cmap=cmap,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
        linecolor="slategray",
        linewidths=0.0,
        clip_on=True,
        ax=ax,
    )
    h.set_facecolor("white")
    h.set_anchor("NW")

    # * adjust ticks
    h.tick_params(
        left=False,
        bottom=False,
        top=False,
        right=False,
        labelbottom=False,
        labelleft=False,
        rotation=0,
        labelsize=3,
        length=0,
        pad=1,
    )
    h.tick_params(axis="y", labelsize=4)
    if orientation == "horizontal":
        h.tick_params(labelsize=1.5)
        h.tick_params(axis="x", rotation=90, labelsize=2)

    # * add colorbar if desired
    if cbar is True and cbar_ax is None:
        cbar = plt.colorbar(
            ax.collections[0],
            shrink=0.2,
            fraction=0.1,
            anchor="NW",
            location=cbar_location,
            use_gridspec=True,
            pad=0,
        )
        cbar.ax.spines["outline"].set_lw(0.4)
        cbar.ax.tick_params(right=False, left=False, labelsize=4, length=0, pad=3)

    # * set title
    if orientation == "vertical":
        ax.set_title(name, fontweight="bold", fontsize=8, pad=2)
    elif orientation == "horizontal":
        ax.set_ylabel(name, fontweight="bold", fontsize=6, labelpad=2)

    # * draw and label wild-type patches
    for j, i in np.asarray(np.where(df_wt)).T:
        if orientation == "horizontal":
            # rotation = 80
            # fontsize = 1
            lw = 0.35
        elif orientation == "vertical":
            # rotation = 0
            # fontsize = 2.5
            lw = 0.75
        h.add_patch(
            Rectangle((i, j), 1, 1, fill=True, color="white", ec="dimgray", lw=lw)
        )
        j += 0.5
        # h.text(
        #     i,
        #     j,
        #     "/",
        #     color="black",
        #     va="center",
        #     fontsize=fontsize,
        #     fontfamily="monospace",
        #     rotation=rotation,
        #     clip_on=True,
        # )
    respine(h)
    # * reformat coordinate labeler
    if orientation == "vertical":

        def format_coord(x, y):
            x = np.floor(x).astype("int")
            y = np.floor(y).astype("int")
            residue = df.columns[x]
            pos = df.index[y]
            fitness_score = df.loc[pos, residue].round(4)
            return f"position: {pos}, residue: {residue}, fitness: {fitness_score}"

    elif orientation == "horizontal":

        def format_coord(x, y):
            x = np.floor(x).astype("int")
            y = np.floor(y).astype("int")
            pos = df.columns[x]
            residue = df.index[y]
            fitness_score = df.loc[residue, pos].round(4)
            return f"position: {pos}, residue: {residue}, fitness: {fitness_score}"

    ax.format_coord = format_coord
    return h


def heatmap_draw(
    data: SequencingData,
    dataset: str,
    gene: Gene,
    read_threshold: int = 20,
    vmin: float = -2.0,
    vmax: float = 2.0,
    fitness_cmap: str = "vlag",
    orientation: str = "horizontal",
) -> matplotlib.figure:
    """
    Draw a heatmap figure of a dataset
    # TODO: Consider re-adding figure to make a missing chart, but perhaps not really necessary

    Parameters
    ----------
    data : SequencingData
        Data from experiment sequencing with count-, enrichment-, and fitness-values
    dataset : str
        Whether to draw a heatmap for raw (log-transformed) counts or fitness values
    gene : Gene
        Gene object that provides residue numbering
    read_threshold : int, optional
        Minimum number of reads for fitness value to be considered valid, by default 20
    vmin : float, optional
        For fitness data, vmin parameter passed to sns.heatmap, by default -2.0
    vmax : float, optional
        For fitness data, vmax parameter passed to sns.heatmap, by default 2.0
    fitness_cmap : str, optional
        Colormap to use for fitness heatmap, by default "vlag"
    orientation : str, optional
        Whether to draw heatmaps vertically or horizontally, by default "horizontal"

    Returns
    -------
    fig : matplotlib.figure
    """
    wt_mask = heatmap_masks(gene)

    # * determine parameters for plotting function based on figure type
    counts_dict = data.counts
    params_counts = {
        "df_dict": counts_dict,
        "num_columns": len(counts_dict),
        "num_rows": 1,
        "suptitle": "Raw counts of mutations ($log_{10}$)",
    }
    if dataset == "counts":
        df_dict, num_columns, num_rows, suptitle = params_counts.values()
    if data.fitness is not None:
        fitness_dict = data.fitness
        params_fitness = {
            "df_dict": fitness_dict,
            "num_columns": len(fitness_dict),
            "num_rows": 1,
            "suptitle": "Fitness values",
        }
        if dataset == "fitness":
            df_dict, num_columns, num_rows, suptitle = params_fitness.values()

    if orientation == "horizontal":
        num_columns, num_rows = num_rows, num_columns
        cbar_location = "bottom"
    else:
        cbar_location = "right"

    fig, axs = plt.subplots(
        num_rows,
        num_columns,
        figsize=(12, 12),
        dpi=300,
        layout="compressed",
        sharex=True,
        sharey=True,
    )
    fig.suptitle(suptitle, fontweight="bold")

    # * plot each data one by one
    for i, sample in enumerate(sorted(counts_dict)):
        # * function-provided styling for heatmaps
        if dataset == "counts":
            data = df_dict[sample]
            heatmap_wrapper(
                data,
                name=sample,
                gene=gene,
                dataset=dataset,
                ax=axs[i],
                orientation=orientation,
            )
        elif dataset == "fitness":
            # * will use filtered data here, but default is to not filter (i.e. read_threshold=1)
            if "UT" in sample:
                continue
            dfs_filtered = filter_fitness_read_noise(
                counts_dict, fitness_dict, read_threshold=read_threshold
            )
            data = dfs_filtered[sample].mask(wt_mask)
            heatmap_wrapper(
                data,
                name=sample,
                gene=gene,
                dataset=dataset,
                ax=axs[i],
                vmin=vmin,
                vmax=vmax,
                fitness_cmap=fitness_cmap,
                orientation=orientation,
            )
    if orientation == "horizontal":
        # * re-size Figure down to height of all subplots combined after plotting
        height = 0
        for ax in fig.axes:
            ax.tick_params(labelleft=True)
            height += (
                ax.get_tightbbox().transformed(fig.dpi_scale_trans.inverted()).height
            )
        fig.axes[0].tick_params(labeltop=True)
        fig.set_figheight(height + 1)
        # * adjust subplot spacing
        pad = fig.get_layout_engine().get()["hspace"] / 2

    elif orientation == "vertical":
        for ax in fig.axes:
            ax.tick_params(labelbottom=True)
        fig.axes[0].tick_params(labelleft=True)
        # * re-size Figure down
        fig.set_figheight(fig.get_tightbbox().height)
        # * adjust sutplob spacing
        pad = fig.get_layout_engine().get()["wspace"] / 2

    # * add colorbar
    cbar = fig.colorbar(
        axs[0].collections[0],
        ax=fig.axes,
        shrink=0.2,
        fraction=0.1,
        pad=pad,
        anchor="NW",
        location=cbar_location,
        use_gridspec=True,
    )
    cbar.ax.spines["outline"].set_lw(0.4)
    cbar.ax.tick_params(right=False, left=False, labelsize=4, length=0, pad=3)
    return fig


def relabel_axis(
    fig: matplotlib.figure, gene: Gene, orientation: str = "horizontal"
) -> None:
    """
    Here we relabel the position-axis of the heatmap figure to use the Ambler numbering system.

    Parameters
    ----------
    fig : matplotlib.figure
        Parent figure of all the heatmap axes
    gene : Gene
        Gene object that holds a numbering system attribute
    orientation : str, optional
        Whether the heatmaps are drawn horizontally or vertically, by default "horizontal"

    Returns
    -------
    None
    """
    # * df for adjusting interactive hover annotations
    df_wt = heatmap_masks(gene)
    rows, cols = df_wt.shape
    if orientation == "vertical":
        fig.axes[0].set_yticklabels(
            np.take(
                gene.ambler_numbering, (fig.axes[0].get_yticks() - 0.5).astype("int64")
            )
        )
        for ax in fig.axes[:-1]:
            data = ax.collections[0].get_array().data.reshape(rows, cols)

            # * adjust jupyter widget interactive hover values
            def format_coord(x, y):
                x = np.floor(x).astype("int")
                y = np.floor(y).astype("int")
                residue = df_wt.columns[x]
                pos = np.take(gene.ambler_numbering, y)
                value = data[y, x].round(4)  # pylint: disable=cell-var-from-loop
                return f"position: {pos}, residue: {residue}, value: {value}"

            ax.format_coord = format_coord
    elif orientation == "horizontal":
        fig.axes[0].set_xticklabels(
            np.take(
                gene.ambler_numbering, (fig.axes[0].get_xticks() - 0.5).astype("int64")
            )
        )
        for ax in fig.axes[:-1]:
            data = ax.collections[0].get_array().data.reshape(rows, cols)

            # * adjust jupyter widget interactive hover values
            def format_coord(x, y):
                x = np.floor(x).astype("int")
                y = np.floor(y).astype("int")
                residue = df_wt.columns[y]
                pos = np.take(gene.ambler_numbering, x)
                value = data[y, x].round(4)  # pylint: disable=cell-var-from-loop
                return f"position: {pos}, residue: {residue}, value: {value}"

            ax.format_coord = format_coord
