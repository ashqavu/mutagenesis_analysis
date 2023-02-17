#!/usr/bin/env python
"""
Utility functions for doing analyses
"""
import re
import warnings

import numpy as np
import pandas as pd
from Bio.Data import IUPACData
from sklearn.mixture import GaussianMixture

from plasmid_map import Gene


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


def get_gaussian_model(df_x: pd.DataFrame, df_y: pd.DataFrame) -> GaussianMixture:
    """
    Use relative fitness values of synonymous mutations to generate a Gaussian model that
    will determine the bounds of significance for fitness effects

    Parameters
    ----------
    df_x : pd.DataFrame
        Relative fitness data for first drug replicate
    df_y : pd.DataFrame
        Relative fitness data for second drug replicate

    Returns
    -------
    gaussian_model : GaussianMixture
    """
    # * select synonymous mutations
    x_syn = df_x["∅"]
    y_syn = df_y["∅"]
    # * build numpy matrix for gaussian model fitting
    X_syn_train = np.column_stack((x_syn.values.flatten(), y_syn.values.flatten()))
    # filter NaN in pairs
    X_syn_train = X_syn_train[~np.isnan(X_syn_train).any(axis=1)]

    # * train model off of synonymous mutation data
    gaussian_model = GaussianMixture(
        n_components=1, covariance_type="full", means_init=[[0, 0]]
    )
    gaussian_model.fit(X_syn_train)
    return gaussian_model


def ellipse_coordinates(covariance: np.ndarray) -> tuple[float, float, float]:
    """
    Given a position and covariance, calculate the boundaries of the ellipse to be drawn

    Parameters
    ----------
    covariance : np.ndarray
        Covariance array

    Returns
    -------
    width, height, angle : tuple[int, int, int]
        Width, height, and angle of the ellipse
    """

    # convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, _ = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    return width, height, angle


def get_ellipses(gaussian_model: GaussianMixture, sigma_cutoff):
    ellipses = {}
    for center, covar in zip(gaussian_model.means_, gaussian_model.covariances_):
        # a loop here in case there's more than one for some reason
        width, height, angle = ellipse_coordinates(covar)
    for n_sigma in range(1, sigma_cutoff + 1):
        width_sigma = n_sigma * width
        height_sigma = n_sigma * height
        ellipses[n_sigma] = [center, width_sigma, height_sigma, angle]
    return ellipses


def determine_significance(df_x, df_y, ellipse):
    center_f, width, height, angle = ellipse
    width_f = width / 2
    height_f = height / 2
    angle_f = np.radians(angle)
    # * find points that are outside the bounds of significance
    def is_outside_ellipse(point):
        # if either fitness value is NaN, discard point
        if np.isnan(point).any():
            return False
        x, y = point
        cos_angle = np.cos(angle_f)
        sin_angle = np.sin(angle_f)
        x_center, y_center = center_f[0], center_f[1]
        x_dist = x - x_center
        y_dist = y - y_center
        a = ((x_dist * cos_angle) + (y_dist * sin_angle)) ** 2
        b = ((x_dist * sin_angle) - (y_dist * cos_angle)) ** 2
        limit = (width_f**2) * (height_f**2)
        return a * (height_f**2) + b * (width_f**2) > limit

    def is_negative_quadrant(point):
        return (np.array(point) <= 0).all()

    def is_positive_quadrant(point):
        return (np.array(point) >= 0).all()

    def is_sign_sensitive(point):
        return is_outside_ellipse(point) and is_negative_quadrant(point)

    def is_sign_resistant(point):
        return is_outside_ellipse(point) and is_positive_quadrant(point)

    # * merge two dataframes element-wise for significance test
    df_xy = pd.concat([df_x, df_y]).groupby(level=0, axis=0).agg(list)

    df_sign_sensitive = df_xy.applymap(is_sign_sensitive)
    df_sign_resistant = df_xy.applymap(is_sign_resistant)

    return df_sign_sensitive, df_sign_resistant


def gaussian_significance(
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    sigma_cutoff: int = 4,
):
    gaussian_model = get_gaussian_model(df_x, df_y)
    ellipses = get_ellipses(gaussian_model, sigma_cutoff=sigma_cutoff)
    significance_ellipse = ellipses[sigma_cutoff]
    df_sign_sensitive, df_sign_resistant = determine_significance(
        df_x, df_y, significance_ellipse
    )
    return df_sign_sensitive, df_sign_resistant, ellipses


def significant_sigma_dfs(
    counts_dict: dict,
    fitness_dict: dict,
    gene: Gene,
    read_threshold: int = 20,
    sigma_cutoff: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts the significant residue positions from the fitness dataframes

    Parameters
    ----------
    counts_dict : dict
        Reference for counts values of all samples
    fitness_dict : dict
        Reference for fitness values of all samples
    gene : Gene
        Gene object for locating wild-type residues
    read_threshold : int, optional
        Minimum number of reads required to be included, by default 20
    sigma_cutoff : int, optional
        How many sigmas away from the synonymous mutation values to use as the
        cutoff for significance, by default 4

    Returns
    -------
    sign_sensitive_dfs, sign_resistant_dfs : tuple[pd.DataFrame, pd.DataFrame]
        Dataframes of boolean values indicating which cells of the table are
        relevant mutations for the drug
    """
    sign_sensitive_dfs = {}
    sign_resistant_dfs = {}
    drugs = set([x.rstrip("1234567890") for x in fitness_dict])
    for drug in drugs:
        x, y = get_pairs(drug, fitness_dict)
        df_x = filter_fitness_read_noise(
            x, counts_dict, fitness_dict, gene, read_threshold=read_threshold
        )
        df_y = filter_fitness_read_noise(
            y, counts_dict, fitness_dict, gene, read_threshold=read_threshold
        )

        sign_sensitive, sign_resistant, _ = gaussian_significance(
            df_x,
            df_y,
            sigma_cutoff=sigma_cutoff,
        )
        sign_sensitive_dfs[drug] = sign_sensitive
        sign_resistant_dfs[drug] = sign_resistant
    return sign_sensitive_dfs, sign_resistant_dfs

def significant_sigma_mutations(
    counts_dict: dict,
    fitness_dict: dict,
    gene,
    read_threshold: int = 20,
    sigma_cutoff: int = 4,
) -> pd.DataFrame:
    cds_translation = gene.cds_translation
    drugs = sorted(set([x.rstrip("1234567890") for x in fitness_dict]))
    sign_sensitive_dfs, sign_resistant_dfs = significant_sigma_dfs(
        counts_dict,
        fitness_dict,
        gene,
        read_threshold=read_threshold,
        sigma_cutoff=sigma_cutoff,
    )

    sample_name = list(counts_dict)[0]
    point_mutations = pd.MultiIndex.from_product(
        [fitness_dict[sample_name].index, fitness_dict[sample_name].columns]
    ).values
    list_all_fitness = []
    for drug in drugs:
        replica_one, replica_two = get_pairs(drug, fitness_dict)
        df1 = fitness_dict[replica_one]
        df2 = fitness_dict[replica_two]

        sign_sensitive = sign_sensitive_dfs[drug]
        sign_resistant = sign_resistant_dfs[drug]

        for position, residue in point_mutations:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                fitness_entry = {
                    "aa_pos": position,
                    "ref_aa": cds_translation[position],
                    "query_aa": residue,
                    "drug": drug,
                    "rel_fitness_1": df1.loc[position, residue],
                    "rel_fitness_2": df2.loc[position, residue],
                    "rel_fitness_mean": np.nanmean(
                        [df1.loc[position, residue], df2.loc[position, residue]]
                    ),
                    "significant": (
                        sign_sensitive.loc[position, residue]
                        | sign_resistant.loc[position, residue]
                    ),
                }
            if fitness_entry["significant"]:
                if sign_sensitive.loc[position, residue]:
                    fitness_entry.update({"type": "sensitive"})
                elif sign_resistant.loc[position, residue]:
                    fitness_entry.update({"type": "resistance"})
            list_all_fitness.append(fitness_entry)

    df_all_fitness_sigma = pd.DataFrame(list_all_fitness)
    return df_all_fitness_sigma