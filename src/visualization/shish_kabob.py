#!/usr/bin/env python
"""
Plot shish-kabob plots for significant mutations found in mutagenesis studies
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

from fitness_analysis import (
    significance_sigma_dfs_1d, significance_sigma_dfs_2d
)
from sequencing_data import SequencingData
from visualization.gaussians import gaussian_drug_2d, gaussian_drug_1d
from visualization.histograms import histogram_fitness_wrapper
from utils.seq_data_utils import heatmap_masks


def shish_kabob_drug(
    drug: str,
    gaussian: str,
    data: SequencingData,
    read_threshold: int = 20,
    sigma_cutoff: int = 4,
    ax: matplotlib.axes.Axes = None,
    orientation: str = "horizontal",
    vmin: float = -1.5,
    vmax: float = 1.5,
    cbar: bool = False,
) -> matplotlib.axes.Axes:
    """
    drug : str
        Name of drug to plot
    gaussian : str
        "nD" dimensions to use for Gaussian significance
    read_threshold : int, optional
        Minimum number of reads required to be included, by default 20
    sigma_cutoff : int, optional
        How many sigmas away from the synonymous mutation values to use as the
        cutoff for significance, by default 4
    ax : matplotlib.axes.Axes, optional
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
    ax : matplotlib.axes.Axes
    """
    gene = data.gene
    if ax is None:
        ax = plt.gca()

    if gaussian == "2D":
        # * find fitness value of greatest magnitude between pair
        replica_one, replica_two = data.get_pairs(drug, data.samples)
        df1 = data.fitness[replica_one]
        df2 = data.fitness[replica_two]
        
        # * find fitness value of greatest magnitude between pair
        df = df1[df1.abs().ge(df2.abs())]
        df.update(df2[df2.abs().ge(df1.abs())])

        (
            significant_sensitive_dfs,
            significant_resistant_dfs
        ) = significance_sigma_dfs_2d(
            data, read_threshold=read_threshold, sigma_cutoff=sigma_cutoff
        )
        
    elif gaussian == "1D":
        df = data.fitness[drug]

        (
            significant_sensitive_dfs,
            significant_resistant_dfs,
        ) = significance_sigma_dfs_1d(
            data, read_threshold=read_threshold, sigma_cutoff=sigma_cutoff
        )
    significant_sensitive = significant_sensitive_dfs[drug]
    significant_resistant = significant_resistant_dfs[drug]

    # * get residue positions with significant mutations
    significant_positions = (
        significant_sensitive.drop("*", axis=1)
        | significant_resistant.drop("*", axis=1)
    ).sum(axis=1) > 0
    significant_positions = significant_positions[significant_positions].index

    # * select only mutations with significant fitness values
    df_masked = df.where(significant_resistant | significant_sensitive)
    df_masked = df_masked.drop("∅", axis=1)

    with sns.axes_style("white"):
        if orientation == "vertical":
            df_masked_plot = df_masked.loc[significant_positions]

            sns.heatmap(
                df_masked_plot,
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                cbar=cbar,
                square=True,
                ax=ax,
                zorder=100,
            )
            ax.yaxis.grid("on")
            ax.set_yticks(
                np.arange(len(significant_positions)) + 0.5,
                np.array(significant_positions) + 1,
                rotation=0,
                ha="center",
                fontsize="xx-small",
            )
            ax.set_xticks([])
            # * add wild-type notations
            # get reference residues
            ref_aas = np.take(gene.cds_translation, significant_positions)
            # iterate over amino acid options (y-axis)
            for y, residue in enumerate(ref_aas):
                # determine x position for text box
                x = df_masked_plot.columns.get_loc(residue)
                ax.add_patch(
                    Rectangle(
                        (x, y),
                        1,
                        1,
                        ec="black",
                        fc="white",
                        fill=True,
                        lw=0.2,
                        zorder=100,
                    )
                )
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    residue,
                    fontsize="xx-small",
                    ha="center",
                    va="center",
                    zorder=101,
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
                        zorder=101,
                    )
            ax.set_title(drug, fontweight="heavy")
            ax.set_anchor("N")

        elif orientation == "horizontal":
            df_masked = df_masked.T
            df_masked_plot = df_masked[significant_positions]

            sns.heatmap(
                df_masked_plot,
                cmap="vlag",
                vmin=vmin,
                vmax=vmax,
                cbar=cbar,
                square=True,
                ax=ax,
                zorder=100,
            )
            ax.xaxis.grid("on")
            ax.set_xticks(
                np.arange(len(significant_positions)) + 0.5,
                np.array(significant_positions) + 1,
                rotation=90,
                ha="center",
                fontsize="xx-small",
            )
            ax.set_yticks([])
            # * add wild-type notations
            # get reference residues
            ref_aas = np.take(gene.cds_translation, significant_positions)
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
                        zorder=100,
                    )
                )
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    residue,
                    fontsize="xx-small",
                    ha="center",
                    va="center",
                    zorder=101,
                )
            for x, pos in enumerate(significant_positions):
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
                        zorder=101,
                    )
            ax.set_ylabel(drug, fontweight="heavy")
        ax.tick_params(bottom=False, left=False, pad=1)
        return ax


def shish_kabob_draw(
    data: SequencingData,
    gaussian: str,
    read_threshold: int = 20,
    sigma_cutoff: int = 4,
    orientation: str = "horizontal",
    vmin: float = -1.5,
    vmax: float = 1.5,
    xlim: tuple[float, float] = (-2.5, 2.5),
    ylim: tuple[float, float] = (-2.5, 2.5),
) -> matplotlib.axes.Axes:
    """
    Draw shish kabob plots and corresponding gaussian scatter plots for all
    samples in dataset

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
    drugs_all = sorted(drug for drug in data.treatments if "UT" not in drug)
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

            if gaussian == "2D":
                pass
                gaussian_drug_2d(
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
                    "2D",
                    data,
                    read_threshold=read_threshold,
                    sigma_cutoff=sigma_cutoff,
                    ax=ax_shish,
                    orientation=orientation,
                    vmin=vmin,
                    vmax=vmax,
                )
            elif gaussian == "1D":
                wt_mask = heatmap_masks(data.gene)
                # get bins for histogram
                df = data.filter_fitness_read_noise(read_threshold=read_threshold)[drug]
                df = df.mask(wt_mask)
                df.name = drug
                values_fitness = df.values
                bins = np.linspace(
                    np.nanmin(values_fitness), np.nanmax(values_fitness), 25
                )
                histogram_fitness_wrapper(df, bins=bins, ax=ax_gauss)
                gaussian_drug_1d(df, ax_gauss, sigma_cutoff=sigma_cutoff)
                ax_gauss.get_legend().remove()

                shish_kabob_drug(
                    drug,
                    "1D",
                    data,
                    read_threshold=read_threshold,
                    sigma_cutoff=sigma_cutoff,
                    ax=ax_shish,
                    orientation=orientation,
                    vmin=vmin,
                    vmax=vmax,
                )
        if gaussian == "1D":
            adjusted_height = ax_shish.get_tightbbox().transformed(
                fig.dpi_scale_trans.inverted()
            ).height * (i + 1)
            fig.set_figheight(adjusted_height)

    return fig
