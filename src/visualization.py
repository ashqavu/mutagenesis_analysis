#!/usr/bin/env python
from Bio.Data import IUPACData
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.offsetbox import AnchoredText
from scipy.stats import norm


def heatmap_table(gene):
    df = pd.DataFrame(
        index=np.arange(len(gene.cds_translation)),
        columns=list(IUPACData.protein_letters + "*∅"),
    )
    return df


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


# //* should do something about the numbering flexibility here
def heatmap_masks(gene):
    df_mask = heatmap_table(gene)
    df_mask_annot = df_mask.copy()
    df_mask_annot.loc[:, :] = ""
    for position, residue in enumerate(gene.cds_translation):
        df_mask_annot.loc[position, residue] = "/"
    df_mask_WT = df_mask_annot == "/"
    return df_mask_WT, df_mask_annot


def heatmap_gridspec_layout(gene, num_plots, orientation="vertical", grid=False):
    columns = list(IUPACData.protein_letters + "*∅")

    tick_positions = np.arange(len(gene.cds_translation), step=15)

    num_rows = 1
    num_columns = num_plots + 1
    height = 25
    width = 2.5 * num_plots + 0.25
    width_ratios = [1] * num_plots + [0.1]
    figsize = (width, height)
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    # transpose the gridspec
    if orientation == "horizontal":
        num_rows, num_columns = num_columns, num_rows
        width, height = height, width
        height_ratios = width_ratios
        fig.set_figheight(height)
        fig.set_figwidth(width)
        gs = fig.add_gridspec(num_rows, num_columns, height_ratios=height_ratios)

    else:
        gs = fig.add_gridspec(num_rows, num_columns, width_ratios=width_ratios)

    axes = []
    for i in range(num_plots):
        if i >= 1:
            axes.append(plt.subplot(gs[i], anchor="NW", sharey=axes[0], sharex=axes[0]))
        else:
            axes.append(plt.subplot(gs[i], anchor="NW"))

    cbar_ax = plt.subplot(gs[num_plots], label="colorbar", aspect=3, anchor="NW")

    # adjust the colorbar shape and adjust ticklabel positions
    if orientation == "horizontal":
        cbar_ax.set_aspect(0.35)
        for i, ax in enumerate(axes):
            fig.canvas.draw()
            ax.set_yticks(np.arange(22), labels=columns, fontsize=6)
            ax.set_xticks(
                tick_positions,
                fontsize=8,
                fontweight="bold",
            )
            ax.tick_params(axis="both", left=False, bottom=False, labelbottom=False)
            if i == 0:
                ax.tick_params(axis="x", labeltop=True, rotation=90)

            # draw grid
            if grid:
                ax.set_yticks(np.arange(23) - 0.5, minor=True)
                ax.set_xticks(np.arange(len(gene.cds_translation) + 1) - 0.5, minor=True)
                ax.tick_params(which="minor", left=False, bottom=False)
                ax.grid(which="minor", color="lightgray")

        cbar_ax.tick_params(left=False, bottom=False, labelleft=False)

    else:
        for i, ax in enumerate(axes):
            # fig.canvas.draw()
            ax.set_yticks(
                tick_positions[::-1],
                fontsize=8,
                fontweight="bold",
            )
            ax.set_xticks(np.arange(22), labels=columns, fontsize=6)
            ax.tick_params(axis="both", left=False, bottom=False, labelleft=False)
            ax.tick_params(axis="x", labelrotation=0, pad=0)
            if i == 0:
                ax.tick_params(axis="y", labelleft=True)

            # draw grid
            # ax.set_xticks(np.arange(23) - 0.5, minor=True)
            # ax.set_yticks(np.arange(len(gene.cds_translation)) - 0.5, minor=True)
            ax.tick_params(which="minor", left=False, bottom=False)
            ax.grid(which="minor", color="lightgray")

        cbar_ax.tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False,
            labelright=True,
        )

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


def heatmap_raw_counts(df, ax=None, cbar_ax=None, orientation="vertical"):
    if ax is None:
        ax = plt.subplot()
    with np.errstate(divide="ignore"):
        df_log_counts = df.where(df.lt(1), np.log10(df))
    if orientation == "horizontal":
        df_log_counts = df_log_counts.T
    im = ax.imshow(df_log_counts, cmap="Blues")

    # add colorbar
    cbar = plt.colorbar(
        im,
        cax=cbar_ax,
        orientation=orientation,
    )
    cbar.ax.tick_params(bottom=False, right=False)
    return ax


def heatmap_fitness(
    gene,
    df,
    ax=None,
    cbar_ax=None,
    cmap="bwr",
    vmin=-2.0,
    vmax=2.0,
    cbar_kws={},
    orientation="vertical",
):
    if ax is None:
        ax = plt.subplot()

    df_mask_WT, df_mask_annot = heatmap_masks(gene)
    df_mask_WT = df_mask_WT.loc[df.index]
    df_mask_annot = df_mask_annot.loc[df.index]

    if orientation == "horizontal":
        df = df.T
        df_mask_WT = df_mask_WT.T
        df_mask_annot = df_mask_annot.T
    im = ax.imshow(df, cmap=cmap, vmin=vmin, vmax=vmax, label="im", zorder=1)
    ax.set_facecolor("black")

    # annotations
    mask = np.ma.masked_where(~df_mask_WT, np.zeros(df_mask_WT.shape))
    # set background for WT to gray
    im_annot = ax.imshow(mask, cmap=colors.ListedColormap(["slategray"]))
    # add slash marks
    for i in range(df_mask_annot.shape[0]):
        for j in range(df_mask_annot.shape[1]):
            if df_mask_annot.iloc[i, j] == "/":
                im_annot.axes.text(
                    j,
                    i,
                    "/",
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=7,
                    clip_on=True,
                )

    # add colorbar
    cbar = plt.colorbar(
        im,
        cax=cbar_ax,
        orientation=orientation,
    )
    cbar.ax.tick_params(bottom=False, right=False)

    return ax


def draw_heatmap_figure(
    gene, SequencingData, figure_type, orientation="vertical", add_cursor=False
):
    treatments = [x for x in SequencingData.treatments if "UT" not in x]
    nPlots = len(treatments)

    fig = heatmap_gridspec_layout(gene, nPlots, orientation=orientation)

    if figure_type == "heatmap_missing":
        plt.close(fig)
        fig = heatmap_gridspec_layout(gene, 1, orientation=orientation)
        for sample, counts in SequencingData.counts.items():
            if "UT" not in sample:
                continue
            else:
                ax = heatmap_missing_mutations(
                    counts,
                    ax=fig.axes[0],
                    cbar_ax=fig.axes[-1],
                    orientation=orientation,
                )

                # add axes title
                if orientation == "horizontal":
                    ax.set_ylabel(
                        sample, fontsize=18, fontweight="semibold", labelpad=0
                    )
                else:
                    ax.set_title(sample, fontsize=18, fontweight="semibold")

    elif figure_type == "heatmap_counts":
        plt.close()
        fig = heatmap_gridspec_layout(gene, nPlots + 1, orientation=orientation)
        fig.suptitle(
            "Raw counts of mutations ($log_{10}$)", fontsize=20, fontweight="bold"
        )
        for i, sample in enumerate(SequencingData.counts):
            df = SequencingData.counts[sample]
            ax = heatmap_raw_counts(
                df, ax=fig.axes[i], cbar_ax=fig.axes[-1], orientation=orientation
            )
            # add axes title
            if orientation == "horizontal":
                ax.set_ylabel(sample, fontsize=18, fontweight="semibold", labelpad=0)
            else:
                ax.set_title(sample, fontsize=18, fontweight="semibold")

    elif figure_type == "heatmap_fitness":
        fig.suptitle("Fitness effects of mutations", fontsize=20, fontweight="bold")
        for i, sample in enumerate(SequencingData.fitness):
            df = SequencingData.fitness[sample]
            ax = heatmap_fitness(
                gene, df, ax=fig.axes[i], cbar_ax=fig.axes[-1], orientation=orientation
            )

            # add axes title
            if orientation == "horizontal":
                ax.set_ylabel(sample, fontsize=18, fontweight="semibold", labelpad=0)
            else:
                ax.set_title(sample, fontsize=18, fontweight="semibold")

            if add_cursor:
                # add a cursor that displays position-mutation on click
                cursor = mplcursors.cursor()

                @cursor.connect("add")
                def on_click(sel):
                    i, j = sel.index
                    arrData = sel.artist.get_array()
                    dValue = round(arrData[i, j], 3)
                    sel.annotation.set_text(f"{df.index[i]}{df.columns[j]}\n[{dValue}]")

    return fig
