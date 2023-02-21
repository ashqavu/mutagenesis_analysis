#!/usr/bin/env python
"""
Suite of functions written for generating figures associated with deep
mutagenesis library selection experiments
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Rectangle, Ellipse
from scipy.stats import norm

from plasmid_map import Gene
from sequencing_data import (
    SequencingData,
    get_pairs,
    match_treated_untreated,
    filter_fitness_read_noise,
    heatmap_masks,
)

from fitness_analysis import gaussian_significance


def respine(ax: matplotlib.axes) -> None:
    """
    Set the edges of the axes to be solid gray

    Parameters
    ----------
        ax : Axes
            Axes with heatmap plot

    Returns
    -------
        None
    """
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_edgecolor("darkslategray")
        spine.set_lw(0.4)


def histogram_mutation_counts(
    data: SequencingData, read_threshold: int = 20
) -> matplotlib.figure:  # pylint: disable=too-many-locals
    """
    Generate Figure of histograms plotting distribution of number of counts
    found per amino acid mutation

    Parameters
    ----------
    data : SequencingData
        Object providing data for number of counts found per sample
    read_threshold : int, optional
        Minimum number of reads for fitness value to be considered valid, by default 1

    Returns
    -------
    fig : matplotlib.figure
        Figure with each sample plotted on a different Subplot
    """
    counts = data.counts
    wt_mask = heatmap_masks(data.gene)
    # num_plots = len(counts)
    # height = num_plots * 1.8
    samples = list(sorted(data.samples))
    num_subplots = len(samples)
    num_rows = num_columns = int(np.round(np.sqrt(num_subplots)))
    if num_subplots / num_rows > num_rows:
        num_columns = num_rows + 1

    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(num_columns * 8, num_rows * 4),
        constrained_layout=True,
        sharey=True,
        sharex=True,
    )
    fig.suptitle("Distribution of counts for all amino acids")

    for i, sample in enumerate(counts):
        # ! wild-type mask
        counts_values = counts[sample].mask(wt_mask)
        # ! these indices are specific to the mature TEM-1 protein
        # ! would need to be changed if you used a different gene
        counts_values = data.counts[sample].loc[23:285].drop(["*", "∅"], axis=1)
        library_size = counts_values.shape[0] * (counts_values.shape[1] - 1)
        num_missing = counts_values.lt(read_threshold).sum().sum()
        pct_missing = num_missing / library_size

        # * all counts are included in histogram and determining mean number of reads
        mean, _ = norm.fit(counts_values)
        log_values = counts_values.where(
            counts_values.lt(1), lambda x: np.log10(x + 1)
        ).values.flatten()

        ax = axes.flat[i]
        sns.histplot(
            log_values,
            bins=40,
            fc="gray",
            ec="black",
            ax=ax,
        )

        ax.set_ylabel("number of amino acid mutations", fontsize=7)
        ax.set_xlabel("counts per amino acid mutation\n($log_{10}(x+1)$)", fontsize=7)

        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.tick_params(direction="in", labelsize=7)
        ax.set_title(sample, fontsize=12, fontweight="bold")

        text_mean = f"below threshold: {num_missing} ({pct_missing:.2%})\nmean of all: {round(mean, 3)}"
        annot_box = AnchoredText(
            text_mean, loc="upper right", pad=0.8, prop=dict(size="large"), frameon=True
        )
        ax.add_artist(annot_box)
    return fig


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
    if dataset == "counts":
        with np.errstate(divide="ignore"):
            df = df.where(df.lt(1), np.log10(df))
            cmap = "Blues"
            vmin = None
            vmax = None
    elif dataset == "fitness":
        cmap = fitness_cmap

    xticklabels, yticklabels = 1, 10
    df_wt = heatmap_masks(gene)

    if orientation == "horizontal":
        df_wt = df_wt.T
        df = df.T
        xticklabels, yticklabels = yticklabels, xticklabels

    h = sns.heatmap(
        df,
        square=True,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cbar=cbar,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linecolor="slategray",
        linewidths=0.0,
        clip_on=True,
        ax=ax,
        cbar_ax=cbar_ax,
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

    # * set title
    if orientation == "vertical":
        ax.set_title(name, fontweight="bold", fontsize=8, pad=2)
    elif orientation == "horizontal":
        ax.set_ylabel(name, fontweight="bold", fontsize=6, labelpad=2)

    # * draw and label wild-type patches
    for j, i in np.asarray(np.where(df_wt)).T:
        if orientation == "horizontal":
            rotation = 80
            fontsize = 1
            lw = 0.35
        elif orientation == "vertical":
            rotation = 0
            fontsize = 2.5
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
    counts_dict = data.counts
    fitness_dict = data.fitness
    wt_mask = heatmap_masks(gene)

    # * determine parameters for plotting function based on figure type
    params_counts = {
        "df_dict": counts_dict,
        "num_columns": len(counts_dict),
        "num_rows": 1,
        "suptitle": "Raw counts of mutations ($log_{10}$)",
    }
    params_fitness = {
        "df_dict": fitness_dict,
        "num_columns": len(fitness_dict),
        "num_rows": 1,
        "suptitle": "Fitness values",
    }
    if dataset == "counts":
        df_dict, num_columns, num_rows, suptitle = params_counts.values()
    elif dataset == "fitness":
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


def histogram_fitness_wrapper(
    df_fitness_sample: pd.DataFrame,
    counts_dict: dict,
    bins: list,
    ax: matplotlib.axes = None,
    read_threshold: int = 20,
) -> None:
    """
    Styler for individual histogram plotting fitness values. Gray bars show
    missense mutation values, green bars show synonymous mutation values, red
    bars show stop mutations values.

    Parameters
    ----------
    df_fitness_sample : pd.DataFrame
        Fitness dataframe to plot
    counts_dict : dict
        Reference for counts values of all samples
    bins : list
        List of bin values
    ax : matplotlib.axes, optional
        AxesSubplot to plot on, by default None
    read_threshold : int, optional
        Minimum number of reads for fitness value to be considered valid, by default 20

    Returns
    -------
    None
    """
    if ax is None:
        ax = plt.gca()
    sample = df_fitness_sample.name
    untreated = match_treated_untreated(sample)
    df_counts_untreated = counts_dict[untreated]
    df_counts_treated = counts_dict[sample]
    # mask so that both treated and untreated are > read threshold
    df_fitness = df_fitness_sample.where(
        df_counts_untreated.ge(read_threshold) & df_counts_treated.ge(read_threshold)
    )
    # select when UT > threshold and treated < threshold (i.e. extinct)
    df_extinct = df_fitness_sample.where(
        df_counts_untreated.ge(read_threshold) & df_counts_treated.lt(read_threshold)
    )

    # ! TEM-1 mat peptide
    df_fitness = df_fitness.loc[23:285]
    df_extinct = df_extinct.loc[23:285]
    # selecting missense mutations
    values_missense = df_fitness.drop(["*", "∅"], axis=1).values.flatten()
    # synonymous mutants
    values_syn = df_fitness["∅"].values.flatten()
    # stop mutations
    values_stop = df_fitness["*"].values.flatten()
    # extinct mutations
    values_extinct = df_extinct.drop(["*", "∅"], axis=1).values.flatten()

    sns.histplot(
        values_missense,
        bins=bins,
        ax=ax,
        color="gray",
        ec="white",
        alpha=0.6,
        label="missense",
        zorder=99,
    )
    sns.histplot(
        values_syn,
        bins=bins,
        ax=ax,
        color="greenyellow",
        ec="white",
        alpha=0.6,
        label="synonymous",
        zorder=101,
    )
    sns.histplot(
        values_stop,
        bins=bins,
        ax=ax,
        color="lightcoral",
        ec="white",
        lw=0.6,
        alpha=0.6,
        label="stop mutations",
        zorder=101,
    )
    sns.histplot(
        values_extinct,
        bins=bins,
        ax=ax,
        color="steelblue",
        ec="white",
        alpha=0.6,
        label="extinct",
        zorder=100,
    )

    ax.set_title(sample, fontweight="bold")
    # ax.set_xlabel("distribution of fitness effects")
    ax.set_ylabel("")


def histogram_fitness_draw(
    data: SequencingData, read_threshold: int = 20
) -> matplotlib.figure:
    """
    Draw a histogram figure for fitness values of a dataset

    Parameters
    ----------
    data : SequencingData
        Data from experiment sequencing with count-, enrichment-, and fitness-values
    read_threshold : int, optional
        Minimum number of reads for fitness value to be considered valid, by default 20

    Returns
    -------
    fig_dfe_all : matplotlib.figure
    """
    counts_dict = data.counts
    fitness_dict = data.fitness
    gene = data.gene
    # ! wild-type mask
    wt_mask = heatmap_masks(gene)

    samples = list(sorted(fitness_dict))
    num_subplots = len(samples)
    num_rows = num_columns = int(np.round(np.sqrt(num_subplots)))
    if num_subplots / num_rows > num_rows:
        num_columns = num_rows + 1

    dfs_fitness_filt = filter_fitness_read_noise(
        counts_dict, fitness_dict, read_threshold=read_threshold
    )

    # get bins for histogram
    values_fitness_all = []
    for value in dfs_fitness_filt.values():
        value = value.mask(wt_mask)
        values_fitness_all.extend(value.values)
    bins = np.linspace(np.nanmin(values_fitness_all), np.nanmax(values_fitness_all), 60)

    # start drawing
    with sns.axes_style("whitegrid"):
        fig_dfe_all, axs = plt.subplots(
            num_rows,
            num_columns,
            figsize=(10, 8),
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        fig_dfe_all.suptitle("fitness effects", fontweight="bold", fontsize="xx-large")
    for i, sample in enumerate(samples):
        if "UT" in sample:
            continue
        # untreated = match_treated_untreated(sample)
        df_fitness_sample = fitness_dict[sample].mask(wt_mask)
        df_fitness_sample.name = sample
        fig_dfe_all.suptitle(
            f"Distribution of fitness effects (min. reads = {read_threshold})",
            fontsize="large",
            fontweight="heavy",
        )
        ax = axs.flat[i]
        histogram_fitness_wrapper(df_fitness_sample, counts_dict, bins, ax=ax)

    fig_dfe_all.legend(
        ["missense", "synonymous", "stop", "extinct"],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        bbox_transform=fig_dfe_all.transFigure,
        frameon=False,
    )
    fig_dfe_all.supxlabel("distribution of fitness effects", fontweight="heavy")
    fig_dfe_all.supylabel("counts", fontweight="heavy")

    return fig_dfe_all


def gaussian_drug(
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

    x, y = get_pairs(drug, data.samples)
    dfs_filtered = filter_fitness_read_noise(
        counts_dict, fitness_dict, read_threshold=read_threshold
    )
    df_x = dfs_filtered[x]
    df_y = dfs_filtered[y]
    df_x = df_x.mask(wt_mask)
    df_y = df_y.mask(wt_mask)
    df_x = df_x.loc[23:285]
    df_y = df_y.loc[23:285]

    sign_sensitive, sign_resistant, ellipses_all = gaussian_significance(
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


def gaussian_replica_pair_draw(
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
        gaussian_drug(
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


def shish_kabob_drug(
    drug: str,
    data: SequencingData,
    read_threshold: int = 20,
    sigma_cutoff: int = 4,
    ax: matplotlib.axes = None,
    orientation: str = "horizontal",
    vmin: float = -1.5,
    vmax: float = 1.5,
    cbar: bool = False,
) -> matplotlib.axes:
    """
    drug : str
        Name of drug to plot
    read_threshold : int, optional
        Minimum number of reads required to be included, by default 20
    sigma_cutoff : int, optional
        How many sigmas away from the synonymous mutation values to use as the
        cutoff for significance, by default 4
    ax : matplotlib.axes, optional
        Axes to draw the plot on, by default None
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
    """
    fitness_dict = data.fitness
    counts_dict = data.counts
    gene = data.gene
    wt_mask = heatmap_masks(gene)

    replica_one, replica_two = get_pairs(drug, data.samples)
    dfs_filtered = filter_fitness_read_noise(
        counts_dict, fitness_dict, read_threshold=read_threshold
    )
    df1 = dfs_filtered[replica_one]
    df2 = dfs_filtered[replica_two]
    df1 = df1.mask(wt_mask)
    df2 = df2.mask(wt_mask)

    sign_sensitive, sign_resistant, _ = gaussian_significance(
        df1,
        df2,
        sigma_cutoff=sigma_cutoff,
    )

    if ax is None:
        ax = plt.gca()
        # * get residue positions with significant mutations
    sign_positions = (
        sign_sensitive.drop("*", axis=1) | sign_resistant.drop("*", axis=1)
    ).sum(axis=1) > 0
    sign_positions = sign_positions[sign_positions].index
    # * find fitness value of greatest magnitude between pair
    df = df1[df1.abs().ge(df2.abs())]
    df.update(df2[df2.abs().ge(df1.abs())])
    # * select only mutations with significant fitness values
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
                    fontsize="xx-small",
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
                    fontsize="xx-small",
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
        return ax


def shish_kabob_draw(
    data: SequencingData,
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
    data : SequencingData
        Data from experiment sequencing with count-, enrichment-, and fitness-values
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
    drugs_all = sorted([drug for drug in data.treatments if "UT" not in drug])
    gridspec_dict = {"wspace": 0, "hspace": 0}
    if orientation == "horizontal":
        num_rows, num_cols = len(drugs_all), 2
        gridspec_dict.update({"width_ratios": [2.5, 1]})
        if sigma_cutoff <= 3:
            gridspec_dict.update({"width_ratios": [4.5, 1]})
        figsize = (7, 17)
    elif orientation == "vertical":
        num_rows, num_cols = 2, len(drugs_all)
        gridspec_dict.update({"height_ratios": [2.5, 1]})
        if sigma_cutoff <= 3:
            gridspec_dict.update({"height_ratios": [4.5, 1]})
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
            f"Significant mutations\nmin. read = {read_threshold}, sigma cutoff = {sigma_cutoff}",
            fontsize="large",
            fontweight="heavy",
        )
        for i, drug in enumerate(drugs_all):
            # * determine subplots for shish kabob and gaussians
            if orientation == "horizontal":
                ax_shish = axs[i, 0]
                ax_gauss = axs[i, 1]
                ax_shish.set_anchor("W")
                ax_gauss.set_anchor("W")
            elif orientation == "vertical":
                ax_shish = axs[0, i]
                ax_gauss = axs[1, i]
                ax_shish.set_anchor("N")
                ax_gauss.set_anchor("N")
            ax_gauss.set_xlabel(f"{drug}1", size="x-small")
            ax_gauss.set_ylabel(f"{drug}2", size="x-small")
            ax_gauss.tick_params(labelsize="xx-small")
            if sigma_cutoff <= 3:
                ax_shish.tick_params(labelsize=4, pad=0)

            gaussian_drug(
                drug,
                data,
                read_threshold=read_threshold,
                sigma_cutoff=sigma_cutoff,
                ax=ax_gauss,
                xlim=xlim,
                ylim=ylim,
            )

            shish_kabob_drug(
                drug,
                data,
                read_threshold=read_threshold,
                sigma_cutoff=sigma_cutoff,
                ax=ax_shish,
                orientation=orientation,
                vmin=vmin,
                vmax=vmax,
            )

    return fig


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
