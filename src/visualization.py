#!/usr/bin/env python
"""
Suite of functions written for generating figures associated with deep
mutagenesis library selection experiments
"""
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.Data import IUPACData
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Rectangle
from scipy.stats import norm

from plasmid_map import Gene
from sequencing_data import SequencingData


def get_pairs(treatment: str, fitness_dict: dict) -> tuple[str, str]:
    """
    Given a drug, extract the replicas from the fitness dict

    Parameters
    ----------
    treatment : str
        Drug to find replicates of
    fitness_dict : dict
        Reference for fitness values of all samples

    Returns
    -------
    replica_one, replica_two : tuple[str, str]
        Strings of replica sample names
    """
    treatment_pair = [drug for drug in fitness_dict if treatment in drug]
    if not treatment_pair:
        raise KeyError(f"No fitness data: {treatment}")
    if len(treatment_pair) > 2:
        raise IndexError("Treatment has more than 2 replicates to compare")
    replica_one, replica_two = treatment_pair[0], treatment_pair[1]
    return replica_one, replica_two


def match_treated_untreated(sample: str) -> str:
    """
    Takes name of treated sample (e.g. CefX3) and matches it to the
    corresponding untreated sample name (UT3) for proper comparisons.

    Parameters
    ----------
    sample : str
        Name of sample

    Returns
    -------
    untreated : str
        Name of corresponding untreated smple
    """
    num = re.sub(r"[A-Za-z]*", "", sample)
    untreated = "UT" + num
    return untreated


def filter_fitness_read_noise(
    treated: str,
    counts_dict: dict,
    fitness_dict: dict,
    gene: Gene,
    read_threshold: int = 1,
) -> pd.DataFrame:
    """
    Takes DataFrames for treated sample and returns a new DataFrame with cells
    with untreated counts under the minimum read threshold filtered out

    Parameters
    ----------
    treated : str
        Name of treated sample
    counts_dict : dict
        Dictionary containing all samples and DataFrames with mutation count values
    fitness_dict : dict
        Dictionary containing all samples and DataFrames with mutation fitness values
    gene : Gene
        Gene object for locating wild-type residues
    read_threshold : int, optional
        Minimum number of reads required to be included, by default 1

    Returns
    -------
    df_treated_filtered : pd.DataFrame
        Fitness table with insufficient counts filtered out
    """
    untreated = match_treated_untreated(treated)
    df_counts_treated = counts_dict[treated]
    df_counts_untreated = counts_dict[untreated]
    df_treated_filtered = fitness_dict[treated].where(
        df_counts_treated.ge(read_threshold)
        & df_counts_untreated.ge(read_threshold)
        & ~heatmap_masks(gene)
    )
    return df_treated_filtered


def heatmap_table(gene: Gene) -> pd.DataFrame:
    """
    Returns DataFrame for plotting heatmaps with position indices and residue
    columns (ACDEFGHIKLMNPQRSTVWY*∅)

    Parameters
    ----------
    gene : Gene
        Gene object with translated protein sequence

    Returns
    -------
    df : pd.DataFrame
        DataFrame of Falses
    """
    df = pd.DataFrame(
        False,
        index=np.arange(len(gene.cds_translation)),
        columns=list(IUPACData.protein_letters + "*∅"),
    )
    return df


def heatmap_masks(gene: Gene) -> pd.DataFrame:
    """
    Returns a bool DataFrame with wild-type cells marked as True for heatmap
    plotting

    Parameters
    ----------
    gene : Gene
        Object providing translated protein sequence

    Returns
    -------
    df_wt : pd.DataFrame
        DataFrame to use for marking wild-type cells on heatmaps
    """
    df_wt = heatmap_table(gene)
    for position, residue in enumerate(gene.cds_translation):
        df_wt.loc[position, residue] = True
    return df_wt


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


def histogram_mutation_counts(sequencing_data: SequencingData) -> matplotlib.figure:
    """
    Generate Figure of histograms plotting distribution of number of counts
    found per amino acid mutation

    Parameters
    ----------
    sequencing_data : SequencingData
        Object providing data for number of counts found per sample

    Returns
    -------
    fig : matplotlib.figure
        Figure with each sample plotted on a different Subplot
    """
    counts = sequencing_data.counts
    num_plots = len(counts)
    height = num_plots * 1.8

    fig, axes = plt.subplots(
        nrows=num_plots, figsize=(5, height), constrained_layout=True, sharey=True
    )
    fig.suptitle("Distribution of counts for all amino acids")

    for i, sample in enumerate(counts):
        # * these indices are specific to the mature TEM-1 protein
        # * would need to be changed if you used a different gene
        counts_values = counts[sample].loc[23:285, :"Y"]
        num_missing = counts_values.lt(1).sum().sum() - counts_values.shape[0]
        with np.errstate(divide="ignore"):
            log_values = counts_values.where(
                counts_values.lt(1), np.log10(counts_values)
            )
        log_values = log_values.where(log_values != 0.01, np.nan).values.flatten()
        # * total number of mutants specific to TEM-1 library
        pct_missing = num_missing / 4997
        ax = axes[i]
        ax.hist(
            log_values,
            bins=40,
            color="gray",
            edgecolor="black",
            range=(np.nanmin(log_values), np.nanmax(log_values)),
        )

        ax.set_ylabel("number of amino acid mutations", fontsize=7)
        ax.set_xlabel("counts per amino acid mutation ($log_{10}$)", fontsize=7)

        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.tick_params(direction="in", labelsize=7)
        ax.set_title(sample, fontsize=12, fontweight="bold")

        counts_values = counts_values.query("@counts_values.ge(1)").values.flatten()
        counts_values = np.extract(np.isfinite(counts_values), counts_values)
        mean, _ = norm.fit(counts_values)
        text_mean = (
            f"missing: {num_missing} ({pct_missing:.2%})\nmean: {round(mean, 3)}"
        )
        annot_box = AnchoredText(
            text_mean, loc="upper right", pad=0.8, prop=dict(size=6), frameon=True
        )
        ax.add_artist(annot_box)
    return fig


def heatmap_missing_mutations(
    df: pd.DataFrame, ax=None, cbar_ax=None, orientation="vertical"
) -> matplotlib.axes:
    """
    # ! Unused function
    Plot a heatmap showing positions in the library where mutants are missing

    Parameters
    ----------
    df : pandas.DataFrame
        Data matrix to be drawn
    ax : AxesSubplot
        Axes to draw the heatmap
    cbar_ax : AxesSubplot
        Axes to draw the colorbar
    orientation : str, optional
        Whether to draw "horizontal" or "vertical", by default "vertical"


    Returns
    -------
    ax : AxesSubplot
    """
    if ax is None:
        ax = plt.subplot()
    # convert data table from integer counts to a binary map
    df_missing = df.ge(5)
    df_missing = df_missing.loc[:, :"Y"]
    if orientation == "horizontal":
        df_missing = df_missing.T

    im = ax.imshow(df_missing, cmap="Blues")

    # add colorbar index
    cbar = plt.colorbar(
        im,
        cax=cbar_ax,
        orientation=orientation,
        boundaries=[0, 0.5, 1],
        ticks=[0.25, 0.75],
    )
    cbar.ax.tick_params(bottom=False, right=False)
    if orientation == "horizontal":
        cbar.ax.set_xticklabels(["missing", "present"])
        cbar.ax.set_aspect(0.08)
    else:
        cbar.ax.set_yticklabels(["missing", "present"], rotation=-90, va="center")
        cbar.ax.set_aspect(12.5)

    return ax


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
        linewidths=0.2,
        clip_on=True,
        ax=ax,
        cbar_ax=cbar_ax,
    )
    h.set_facecolor("black")
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
        h.add_patch(Rectangle((i, j), 1, 1, fill=True, color="slategray", ec=None))
        j += 0.5
        if orientation == "horizontal":
            rotation = 90
            fontsize = 2
        elif orientation == "vertical":
            rotation = 0
            fontsize = 4
        h.text(
            i,
            j,
            "/",
            color="white",
            va="center",
            fontsize=fontsize,
            fontfamily="monospace",
            rotation=rotation,
            clip_on=True,
        )
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
    counts_dict: dict,
    fitness_dict: dict,
    dataset: str,
    gene: Gene,
    read_threshold: int = 1,
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
    counts_dict : dict
        Reference with counts dataframes for all samples
    fitness_dict : dict
        Reference with fitness dataframes for all samples
    dataset : str
        Whether to draw a heatmap for raw (log-transformed) counts or fitness values
    gene : Gene
        Gene object that provides residue numbering
    read_threshold : int
        Minimum number of reads for fitness value to be considered valid, by default 1
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
        # * will use filtered data here, but default is to not filter (i.e. read_threshold=1)
        df_dict = {
            key: filter_fitness_read_noise(
                key, counts_dict, fitness_dict, gene, read_threshold=read_threshold
            )
            for key in sorted(fitness_dict)
        }

    if orientation == "horizontal":
        num_columns, num_rows = num_rows, num_columns
        cbar_location = "bottom"
    else:
        cbar_location = "right"

    fig, axs = plt.subplots(
        num_rows,
        num_columns,
        figsize=(5, 12),
        dpi=300,
        layout="compressed",
        sharex=True,
        sharey=True,
    )
    fig.suptitle(suptitle, fontweight="bold")

    # * plot each data one by one
    for i, sample in enumerate(sorted(df_dict)):
        data = df_dict[sample]
        # * function-provided styling for heatmaps
        if dataset == "counts":
            heatmap_wrapper(
                data,
                name=sample,
                gene=gene,
                dataset=dataset,
                ax=axs[i],
                orientation=orientation,
            )
        elif dataset == "fitness":
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
                value = data[y, x].round(4)
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
                value = data[y, x].round(4)
                return f"position: {pos}, residue: {residue}, value: {value}"

            ax.format_coord = format_coord


def histogram_fitness_wrapper(
    sample: str, fitness_dict: dict, bins: list, ax: matplotlib.axes = None
) -> None:
    """
    Styler for individual histogram plotting fitness values. Gray bars show
    missense mutation values, green bars show synonymous mutation values, red
    bars show stop mutations values.

    Parameters
    ----------
    sample : str
        Sample to plot
    fitness_dict : dict
        Fitness DataFrames for all samples
    bins : list
        List of bin values
    ax : matplotlib.axes, optional
        AxesSubplot to plot on, by default None

    Returns
    -------
    None
    """
    if ax is None:
        ax = plt.gca()
    df_fitness = fitness_dict[sample]
    # selecting missense mutantions
    values_missense_filtered = df_fitness.drop(["*", "∅"], axis=1).values.flatten()
    # synonymous mutants
    values_syn_filtered = df_fitness["∅"].values.flatten()
    # stop mutations
    values_stop_filtered = df_fitness["*"].values.flatten()

    sns.histplot(
        values_missense_filtered,
        bins=bins,
        ax=ax,
        color="gray",
        label="missense mutations",
    )
    sns.histplot(
        values_syn_filtered,
        bins=bins,
        ax=ax,
        color="palegreen",
        alpha=0.6,
        label="synonymous mutations",
    )
    sns.histplot(
        values_stop_filtered,
        bins=bins,
        ax=ax,
        color="lightcoral",
        alpha=0.6,
        label="stop mutations",
    )

    ax.set_title(sample, fontweight="bold")
    ax.set_xlabel("distribution of fitness effects")
    ax.set_ylabel("counts", weight="bold")


def histogram_fitness_draw(
    counts_dict: dict, fitness_dict: dict, gene: Gene, read_threshold: int = 1
) -> matplotlib.figure:
    """
    Draw a histogram figure for fitness values of a dataset

    Parameters
    ----------
    counts_dict : dict
        DataFrames of count values for all samples
    fitness_dict : dict
        DataFrames of count values for all samples
    gene : Gene
        Object for locating wild-type residues
    read_threshold : int, optional
        Minimum number of reads for fitness value to be considered valid, by default 1

    Returns
    -------
    fig_dfe_all : matplotlib.figure
    """
    samples = list(sorted(fitness_dict))
    num_subplots = len(samples)
    num_rows = num_columns = int(np.round(np.sqrt(num_subplots)))
    if num_subplots / num_rows > num_rows:
        num_columns = num_rows + 1

    fitness_dict_filter = {
        sample: filter_fitness_read_noise(
            sample, counts_dict, fitness_dict, gene, read_threshold=read_threshold
        )
        for sample in samples
    }
    values_fitness_all = np.concatenate(
        [fitness_dict_filter[sample] for sample in samples]
    )
    bins = np.linspace(np.nanmin(values_fitness_all), np.nanmax(values_fitness_all), 51)
    with sns.axes_style("whitegrid"):
        fig_dfe_all, axes = plt.subplots(
            num_rows,
            num_columns,
            figsize=(10, 8),
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        for i, sample in enumerate(samples):
            ax = axes.flat[i]
            histogram_fitness_wrapper(sample, fitness_dict_filter, bins, ax=ax)
        fig_dfe_all.get_layout_engine().set(hspace=0.1)
    return fig_dfe_all
