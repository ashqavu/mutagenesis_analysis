#!/usr/bin/env python

import matplotlib
import numpy as np

from plasmid_map import Gene
from utils import heatmap_masks

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
