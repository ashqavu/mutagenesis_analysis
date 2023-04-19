import re

import numpy as np
import pandas as pd
from Bio.Data import IUPACData

from plasmid_map import Gene

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
