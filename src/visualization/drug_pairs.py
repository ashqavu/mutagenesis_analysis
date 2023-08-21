#!/usr/bin/env python
"""
Plot drug-by-drug relations of significant mutations found in mutagenesis studies
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fitness_analysis import significance_sigma_dfs_1d, significance_sigma_dfs_2d
from sequencing_data import SequencingData
from utils.seq_data_utils import heatmap_masks


def drug_pair(
    drug1: str,
    drug2: str,
    data: SequencingData,
    gaussian: str,
    ax: matplotlib.axes.Axes = None,
    read_threshold: int = 20,
    sigma_cutoff: int = 4,
    xlim: tuple[float, float] = (-2.5, 2.5),
    ylim: tuple[float, float] = (-2.5, 2.5),
    use_synonymous: bool = True,
) -> None:
    """
    Find common significant resistance/sensitivity mutations between two different drugs

    Parameters
    ----------
    drug1 : str
        First drug
    drug2 : str
        Second drug
    data : SequencingData
        Data from experiment sequencing with count-, enrichment-, and fitness-values
    gaussian : str
        "nD" dimensions to use for Gaussian significance
    ax : matplotlib.axes.Axes, optional
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
    use_synonymous : bool, optional
        Whether to build a 1-D model using just the synonymous mutations or not, by default True
    """
    gene = data.gene
    fitness_filtered_dfs = data.filter_fitness_read_noise(read_threshold=read_threshold)

    if ax is None:
        ax = plt.gca()
    wt_mask = heatmap_masks(gene)
    # * get cells of significant mutations
    if gaussian == "2D":

        # drug 1
        drug1_x, drug1_y = data.get_pairs(drug1, data.samples)
        df1_x = fitness_filtered_dfs[drug1_x].mask(wt_mask)
        df1_y = fitness_filtered_dfs[drug1_y].mask(wt_mask)
        # drug 2
        drug2_x, drug2_y = data.get_pairs(drug2, data.samples)
        df2_x = fitness_filtered_dfs[drug2_x].mask(wt_mask)
        df2_y = fitness_filtered_dfs[drug2_y].mask(wt_mask)

        # * find mean of fitness values between replicates
        drug1_df = pd.concat([df1_x, df1_y]).groupby(level=0, axis=0).agg(np.mean)
        drug2_df = pd.concat([df2_x, df2_y]).groupby(level=0, axis=0).agg(np.mean)

        _, significant_sensitive_dfs, significant_resistant_dfs = significance_sigma_dfs_2d(
            data, read_threshold=read_threshold, sigma_cutoff=sigma_cutoff
        )

    elif gaussian == "1D":
        drug1_df = fitness_filtered_dfs[drug1].mask(wt_mask)
        drug2_df = fitness_filtered_dfs[drug2].mask(wt_mask)

        _, significant_sensitive_dfs, significant_resistant_dfs = significance_sigma_dfs_1d(
            data, read_threshold=read_threshold, sigma_cutoff=sigma_cutoff, use_synonymous=use_synonymous
        )
    drug1_significant_sensitive = significant_sensitive_dfs[drug1]
    drug2_significant_sensitive = significant_sensitive_dfs[drug2]

    drug1_significant_resistance = significant_resistant_dfs[drug1]
    drug2_significant_resistance = significant_resistant_dfs[drug2]

    # * build numpy matrix of all points for plotting
    X = np.column_stack((drug1_df.values.flatten(), drug2_df.values.flatten()))
    X = X[np.isfinite(X).all(axis=1)]

    # * 1-D scatterplots
    # * all mutations
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        zorder=-1,
        ax=ax,
        plotnonfinite=False,
        color="gray",
        lw=2,
        s=5
    )
    # * sensitive mutations
    # drug 1
    ax = sns.scatterplot(
        x=drug1_df[drug1_significant_sensitive].values.flatten(),
        y=drug2_df[drug1_significant_sensitive].values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="dodgerblue",
        lw=2,
        s=5
    )
    # drug 2
    ax = sns.scatterplot(
        x=drug1_df[drug2_significant_sensitive].values.flatten(),
        y=drug2_df[drug2_significant_sensitive].values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="dodgerblue",
        lw=2,
        s=5
    )
    # drug1-drug2 shared sensitive mutations
    shared_sensitive_1 = drug1_df.where(
        drug1_significant_sensitive & drug2_significant_sensitive
    )
    shared_sensitive_2 = drug2_df.where(
        drug1_significant_sensitive & drug2_significant_sensitive
    )
    X_shared_sensitive = np.column_stack((shared_sensitive_1.values.flatten(), shared_sensitive_2.values.flatten()))
    X_shared_sensitive = X_shared_sensitive[np.isfinite(X_shared_sensitive).all(axis=1)]
    sns.scatterplot(
        x=X_shared_sensitive[:, 0],
        y=X_shared_sensitive[:, 1],
        ax=ax,
        plotnonfinite=False,
        color="mediumblue",
        lw=2,
        s=5,
        marker="D"
    )


    # * resistance mutations
    # drug 1
    ax = sns.scatterplot(
        x=drug1_df[drug1_significant_resistance].values.flatten(),
        y=drug2_df[drug1_significant_resistance].values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="lightcoral",
        lw=2,
        s=5
    )
    # drug 2
    ax = sns.scatterplot(
        x=drug1_df[drug2_significant_resistance].values.flatten(),
        y=drug2_df[drug2_significant_resistance].values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="lightcoral",
        lw=2,
        s=5
    )
    # drug1-drug2 shared resistance mutations
    shared_resistance_1 = drug1_df.where(
        drug1_significant_resistance & drug2_significant_resistance
    )
    shared_resistance_2 = drug2_df.where(
        drug1_significant_resistance & drug2_significant_resistance
    )
    X_shared_resistance = np.column_stack((shared_resistance_1.values.flatten(), shared_resistance_2.values.flatten()))
    X_shared_resistance = X_shared_resistance[np.isfinite(X_shared_resistance).all(axis=1)]
    sns.scatterplot(
        x=X_shared_resistance[:, 0],
        y=X_shared_resistance[:, 1],
        ax=ax,
        plotnonfinite=False,
        color="firebrick",
        lw=2,
        s=5,
        marker="D"
    )

    ax.plot([-4, 4], [-4, 4], ":", color="gray", alpha=0.5, zorder=0)
    ax.plot([0, 0], [-4, 4], "-", color="gray", alpha=0.5, lw=1, zorder=0)
    ax.plot([-4, 4], [0, 0], "-", color="gray", alpha=0.5, lw=1, zorder=0)
    ax.set(xlim=xlim, ylim=ylim)
    ax.tick_params(left=False, bottom=False, labelsize="xx-small")
    ax.set_anchor("NW")
    ax.set_aspect("equal")
    ax.set_title(
        f"min. reads: {read_threshold}\nsigma cutoff: {sigma_cutoff}",
        fontsize="x-large",
        fontweight="heavy",
    )
    return ax


def drug_pairs_draw(
    data: SequencingData,
    gaussian: str,
    read_threshold: int = 20,
    sigma_cutoff: int = 4,
    xlim: tuple[float, float] = (-2.5, 2.5),
    ylim: tuple[float, float] = (-2.5, 2.5),
    use_synonymous: bool = True
):
    """
    Find common significant resistance/sensitivity mutations between two different drugs
    for all drug combinations

    Parameters
    ----------
    data : SequencingData
        Data from experiment sequencing with count-, enrichment-, and fitness-values
    gaussian : str
        "nD" dimensions to use for Gaussian significance
    read_threshold : int, optional
        Minimum number of reads required to be included, by default 20
    sigma_cutoff : int, optional
        How many sigmas away from the synonymous mutation values to use as the
        cutoff for significance, by default 4
    xlim : tuple[float, float], optional
        X-axis limits of figure, by default (-2.5, 2.5)
    ylim : tuple[float, float], optional
        y-axis limits of figure, by default (-2.5, 2.5)
    use_synonymous : bool, optional
        Whether to build a 1-D model using just the synonymous mutations or not, by default True
    """
    drugs_all = sorted(drug for drug in data.treatments if "UT" not in drug)
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
                try:
                    ax = axs[ax_row - 1, ax_col]
                except TypeError:
                    ax = axs
                drug_pair(
                    drug_x,
                    drug_y,
                    data,
                    gaussian,
                    ax=ax,
                    read_threshold=read_threshold,
                    sigma_cutoff=sigma_cutoff,
                    xlim=xlim,
                    ylim=ylim,
                    use_synonymous=use_synonymous
                )
                if ax_col == 0:
                    ax.set_ylabel(drug_y)
                if ax_row == len(drugs_all) - 1:
                    ax.set_xlabel(drug_x)
            elif ax_row > 0 and (ax_col < len(drugs_all) - 1):
                ax = axs[ax_row - 1, ax_col]
                ax.set_in_layout(False)
                ax.remove()
    for ax in fig.axes:
        ax.set_title("")
    fig.suptitle(
        f"min. reads: {read_threshold}\nsigma cutoff: {sigma_cutoff}",
        fontsize="large",
        fontweight="heavy",
    )
    return fig
