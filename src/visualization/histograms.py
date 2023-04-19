#!/usr/bin/env python
"""
Histogram plots generated for mutagenesis studies
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from scipy.stats import norm

from fitness_analysis import build_gaussian_model_1d, determine_significance_1d
from sequencing_data import SequencingData
from utils.seq_data_utils import (
    heatmap_masks,
)


def histogram_mutation_counts(  # pylint: disable=too-many-locals
    data: SequencingData, read_threshold: int = 20
) -> matplotlib.figure:
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
    fig.suptitle(f"Distribution of counts for all amino acids (min. reads = {read_threshold})", fontsize="xx-large")

    for i, sample in enumerate(counts):
        # ! wild-type mask
        counts_values = counts[sample].mask(wt_mask)
        # ! these indices are specific to the mature TEM-1 protein
        # ! would need to be changed if you used a different gene
        # counts_values = data.counts[sample].loc[23:285].drop(["*", "∅"], axis=1)
        counts_values = data.counts[sample][:-1].drop(["*", "∅"], axis=1)
        library_size = counts_values.shape[0] * (counts_values.shape[1] - 1)
        num_missing = counts_values.lt(read_threshold).sum().sum()
        pct_missing = num_missing / library_size

        # * all counts are included in histogram and determining mean number of reads
        mean = np.nanmean(counts_values)
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

        ax.set_ylabel("")
        # ax.set_xlabel("counts per amino acid mutation\n($log_{10}(x+1)$)", fontsize=7)

        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.tick_params(direction="in", labelsize=7)
        ax.set_title(sample, fontsize=12, fontweight="bold")

        text_mean = f"below threshold (min. {read_threshold}): {num_missing} ({pct_missing:.2%})\nmean of all: {round(mean, 3)}"
        annot_box = AnchoredText(
            text_mean, loc="upper right", pad=0.8, prop=dict(size="large"), frameon=True
        )
        ax.add_artist(annot_box)
    fig.supylabel("num of amino acid mutations", fontsize="large")
    fig.supxlabel("counts per amino acid mutation\n($log_{10}(x+1)$)", fontsize="large")
    return fig


def histogram_fitness_wrapper(
    df_fitness: pd.DataFrame, bins: list, ax: matplotlib.axes = None
) -> None:
    """
    Styler for individual histogram plotting fitness values. Gray bars show
    missense mutation values, green bars show synonymous mutation values, red
    bars show stop mutations values.

    Parameters
    ----------
    df_fitness : pd.DataFrame
        Fitness dataframe to plot
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
    sample = df_fitness.name

    # ! TEM-1 mat peptide
    # df_fitness = df_fitness.loc[23:285]
    # * missense mutations
    # values_missense = df_fitness.drop(["*", "∅"], axis=1).values.flatten()
    # * synonymous mutants
    values_syn = df_fitness["∅"].values.flatten()
    # * stop mutations
    values_stop = df_fitness["*"].values.flatten()

    sns.histplot(
        df_fitness.values.flatten(),
        bins=bins,
        ax=ax,
        color="gray",
        ec="white",
        alpha=0.6,
        label="all",
        zorder=98
    )
    # sns.histplot(
    #     values_missense,
    #     bins=bins,
    #     ax=ax,
    #     color="gray",
    #     ec="white",
    #     alpha=0.6,
    #     label="missense",
    #     zorder=99,
    # )
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
    # sns.histplot(
    #     values_extinct,
    #     bins=bins,
    #     ax=ax,
    #     color="steelblue",
    #     ec="white",
    #     alpha=0.6,
    #     label="extinct",
    #     zorder=100,
    # )

    ax.set_title(sample, fontweight="bold")
    # ax.set_xlabel("distribution of fitness effects")
    ax.set_ylabel("")


def histogram_fitness_draw(
    data: SequencingData,
    read_threshold: int = 20,
    gaussian: bool = False,
    sigma_cutoff: int = 3,
) -> matplotlib.figure:
    """
    Draw a histogram figure for fitness values of a dataset

    Parameters
    ----------
    data : SequencingData
        Data from experiment sequencing with count-, enrichment-, and fitness-values
    read_threshold : int, optional
        Minimum number of reads for fitness value to be considered valid, by default 20
    gaussian: bool, optional

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

    dfs_fitness_filt = data.filter_fitness_read_noise(
        counts_dict, fitness_dict, read_threshold=read_threshold
    )

    # get bins for histogram
    values_fitness_all = []
    for value in dfs_fitness_filt.values():
        value = value.mask(wt_mask)
        values_fitness_all.extend(value.values)
    bins = np.linspace(np.nanmin(values_fitness_all), np.nanmax(values_fitness_all), 40)

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
        fig_dfe_all.suptitle(
            f"Distribution of fitness effects (min. reads = {read_threshold})",
            fontsize="large",
            fontweight="heavy",
        )
    for i, sample in enumerate(samples):
        if "UT" in sample:
            continue
        df_fitness_sample = dfs_fitness_filt[sample].mask(wt_mask)
        df_fitness_sample.name = sample
        ax = axs.flat[i]
        histogram_fitness_wrapper(df_fitness_sample, bins, ax=ax)
        legend_labels = ["all mutations", "synonymous", "stop"]
        if gaussian is True:
            mu, std = build_gaussian_model_1d(df_fitness_sample)
            left_bound = mu - (sigma_cutoff * std)
            right_bound = mu + (sigma_cutoff * std)
            ax.axvline(left_bound, linestyle="dashed", color="blue")
            ax.axvline(right_bound, linestyle="dashed", color="red")
            legend_labels = [f"-{sigma_cutoff}$\sigma$", f"+{sigma_cutoff}$\sigma$", "all mutations", "synonymous", "stop"]

    fig_dfe_all.legend(
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        bbox_transform=fig_dfe_all.transFigure,
        frameon=False,
    )
    fig_dfe_all.supxlabel("distribution of fitness effects", fontweight="heavy")
    fig_dfe_all.supylabel("counts", fontweight="heavy")

    return fig_dfe_all
