#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.Data import IUPACData
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Rectangle
from scipy.stats import norm


def heatmap_table(gene):
    df = pd.DataFrame(
        False,
        index=np.arange(len(gene.cds_translation)),
        columns=list(IUPACData.protein_letters + "*∅"),
    )
    return df


# //* should do something about the numbering flexibility here
def heatmap_masks(gene):
    df_wt = heatmap_table(gene)
    for position, residue in enumerate(gene.cds_translation):
        df_wt.loc[position, residue] = True
    return df_wt


def respine(ax):
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


def histogram_mutation_counts(SequencingData):
    counts = SequencingData.counts
    num_plots = len(counts)
    height = num_plots * 1.8

    fig, axes = plt.subplots(
        nrows=num_plots, figsize=(5, height), constrained_layout=True, sharey=True
    )
    fig.suptitle("Distribution of counts for all amino acids")

    for i, sample in enumerate(counts):
        # * // these indices are specific to the mature TEM-1 protein
        # * // would need to be changed if you used a different gene
        counts_values = counts[sample].loc[23:285, :"Y"]
        num_missing = counts_values.lt(1).sum().sum() - counts_values.shape[0]
        with np.errstate(divide="ignore"):
            log_values = counts_values.where(
                counts_values.lt(1), np.log10(counts_values)
            )
        log_values = log_values.where(log_values != 0.01, np.nan).values.ravel()
        # * // total number of mutants specific to TEM-1 library
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

        counts_values = counts_values.query("@counts_values.ge(1)").values.ravel()
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


def heatmap_missing_mutations(df, ax=None, cbar_ax=None, orientation="vertical"):
    """
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

    # set title

    return ax


def heatmap_wrapper(df: pd.DataFrame, dataset: str, ax=None, cbar=False, cbar_ax=None, cbar_kws=None, vmin=-2, vmax=2):
    """
    Function wrapper for preferred heatmap aesthetic settings

    Parameters
    ----------
    df : pandas.DataFrame
        Matrix to be plotted
    ax : matplotlib.Axes, optional
        Axes on which to draw the data, by default None
    cbar : bool, optional
        Whether to draw a colorbar or not, by default False
    cbar_ax : matplotlib.Axes, optional
        Axes on which to draw the colorbar, by default None
    cbar_kws : _type_, optional
        *kwargs passed to matplotlib.colorbar(), by default None

    Returns
    -------
    h : matplotlib.Axes
        Axes object with the heatmap
    """
    cbar_kws_dict = {"use_gridspec": True}
    if cbar_kws:
        cbar_kws_dict.update(cbar_kws)
    if dataset == "counts":
        with np.errstate(divide="ignore"):
            df = df.where(df.lt(1), np.log10(df))
            cmap = "Blues"
            vmin = None
            vmax = None
    elif dataset == "fitness":
        cmap = "vlag"
    h = sns.heatmap(
        df,
        square=True,
        xticklabels=10,
        yticklabels=1,
        cbar=cbar,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linecolor="slategray",
        edgecolor="darkslategray",
        linewidths=0.2,
        clip_on=False,
        ax=ax,
        cbar_ax=cbar_ax,
        cbar_kws=cbar_kws_dict,
        facecolor="black"
    )
    h.tick_params(axis="x", labelrotation=90, labelbottom=False)
    h.tick_params(axis="y", labelrotation=0)
    respine(h)
    return h


def heatmap_draw(df_dict: dict, dataset: str, gene, vmin=-2, vmax=2):
    """
    Draw a heatmap of a dataset
    # TODO: consider re-adding figure to make a missing chart, but perhaps not really necessary
    # TODO: re-adjust for option of plotting vertically
    # TODO: add annotations for wild-type residues

    Parameters
    ----------
    df_dict : dict
        Sample names with datatables
    dataset : str
        Whether to draw a heatmap for raw (log-transformed) counts or fitness data
    vmin : int, optional
        For fitness data, vmin parameter passed to sns.heatmap, by default -2
    vmax : int, optional
        For fitness data, vmax parameter passed to sns.heatmap, by default 2

    Returns
    -------
    fig
        matplotlib.Figure
    """
    df_wt = heatmap_masks(gene)
    with plt.style.context("heatmap.mplstyle"):
        num_plots = len(df_dict)
        num_rows = num_plots + 1
        num_columns = 1
        height_ratios = [1] * num_plots + [0.1]
        
        fig = plt.figure(constrained_layout=True)
        if dataset == "counts":
            fig.suptitle("Raw counts of mutations ($log_{10}$)")
        elif dataset == "fitness":
            fig.suptitle("Fitness values")
        
        fig.set_figwidth(5)
        gs = fig.add_gridspec(num_rows, num_columns, height_ratios=height_ratios)
        cbar_ax = plt.subplot(gs[-1], aspect=0.3, anchor="NW", label="colorbar")
        
        # * plot each data one by one
        for i, (sample, data) in enumerate(df_dict.items()):
            ax = plt.subplot(gs[i], label=sample, anchor="NW")
            # * function-provided styling for heatmap
            if dataset == "counts":
                # ! watch for transposed data that i need to fix later
                heatmap_wrapper(data.T, dataset=dataset, ax=ax, cbar=True, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"})
            elif dataset =="fitness":
                heatmap_wrapper(data.T, dataset=dataset, ax=ax, cbar=True, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"}, vmin=vmin, vmax=vmax)
            ax.set_ylabel(sample)
            # * add x-axis (position) labels to top subplot
            if i == 0:
                ax.tick_params(axis="x", labeltop=True)
            # * share the x- and y-axis of the data plots
            elif i >= 1:
                ax.sharex(fig.axes[1])
                ax.sharey(fig.axes[1])
            # * draw and label wild-type patches
            for i, j in np.asarray(np.nonzero(df_wt.values)).T: # ! transposed
                ax.add_patch(Rectangle((i, j), 1, 1, fill=True, color="slategray", ec=None))
                ax.text(i, j + 0.5, "/", color="white", va="center", fontsize=1, rotation=90)
            # * reformat coordinate labeler
            def format_coord(x, y):
                x = np.floor(x).astype("int")
                y = np.floor(y).astype("int")
                # ! watch figure orientation
                pos = df_wt.T.columns[x]
                residue = df_wt.T.index[y]
                fitness_score = data.T.loc[residue, pos].round(4)
                return f"position: {pos}, residue: {residue}, fitness: {fitness_score}"
            ax.format_coord = format_coord
        # * adjust colorbar aesthetics
        cbar_ax.spines["outline"].set_visible(True)
        cbar_ax.spines["outline"].set_lw(0.4)
        # * calculate the minimum figure height that keeps aspect ratio of heatmaps
        height = 0
        for ax in fig.axes:
            height += ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).height + 0.1
        fig.set_figheight(height)
    return fig
