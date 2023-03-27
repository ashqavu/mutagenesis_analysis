import re

import numpy as np
import pandas as pd
from Bio.Data import IUPACData

from plasmid_map import Gene


def get_pairs(treatment: str, samples: list) -> tuple[str, str]:
    """
    Given a drug, extract the replicas from the list of samples

    Parameters
    ----------
    treatment : str
        Drug to find replicates of
    samples : list
        Reference for fitness values of all samples

    Returns
    -------
    replica_one, replica_two : tuple[str, str]
        Strings of replica sample names
    """
    treatment_pair = [sample for sample in samples if treatment in sample]
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
    r = re.compile(r"_(\d+)")
    num = r.findall(sample)[0]
    untreated = f"UT_{num}"
    return untreated


def filter_fitness_read_noise(
    counts_dict: dict,
    fitness_dict: dict,
    read_threshold: int = 20,
) -> dict:
    """
    Takes DataFrames for treated sample and returns a new DataFrame with cells
    with untreated counts under the minimum read threshold filtered out

    Parameters
    ----------
    counts_dict : dict
        Reference with counts dataframes for all samples
    fitness_dict : dict
        Reference with fitness dataframes for all samples
    read_threshold : int, optional
        Minimum number of reads required to be included, by default 20

    Returns
    -------
    df_treated_filtered : dict
        Fitness tables with insufficient counts filtered out
    """
    dfs_filtered = {}
    for sample in sorted(fitness_dict):
        untreated = match_treated_untreated(sample)
        df_counts_untreated = counts_dict[untreated]
        df_counts_sample = counts_dict[sample]
        df_fitness_sample = fitness_dict[sample]
        dfs_filtered[sample] = df_fitness_sample.where(
            df_counts_sample.ge(read_threshold) & df_counts_untreated.ge(read_threshold)
        )
    return dfs_filtered


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
