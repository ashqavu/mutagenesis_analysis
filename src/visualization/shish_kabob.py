#!/usr/bin/env python
"""
Plot shish-kabob plots for significant mutations found in mutagenesis studies
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

from fitness_analysis import significant_sigma_dfs_2d
from sequencing_data import SequencingData
from utils.seq_data_utils import heatmap_masks
from visualization.gaussians import gaussian_drug_2d


def shish_kabob_drug(
    drug: str,
    gaussian: str,
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
    gaussian : str
        "nD" dimensions to use for Gaussian significance
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

    if ax is None:
        ax = plt.gca()

    dfs_filtered = data.filter_fitness_read_noise(
        counts_dict, fitness_dict, read_threshold=read_threshold
    )
    
    if gaussian == "2D":
        replica_one, replica_two = data.get_pairs(drug, data.samples)
        df1 = dfs_filtered[replica_one]
        df2 = dfs_filtered[replica_two]
        df1 = df1.mask(wt_mask)
        df2 = df2.mask(wt_mask)

        sign_sensitive, sign_resistant, _ = gaussian_significance_2d(
            df1,
            df2,
            sigma_cutoff=sigma_cutoff,
        )

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
    
    elif gaussian == "1D":
        sign_sensitive, sign_resistant = 

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
                data,
                read_threshold=read_threshold,
                sigma_cutoff=sigma_cutoff,
                ax=ax_shish,
                orientation=orientation,
                vmin=vmin,
                vmax=vmax,
            )

    return fig
