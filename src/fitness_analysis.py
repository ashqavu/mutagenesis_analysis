#!/usr/bin/env python
"""
Utility functions for doing analyses
"""
import re
from typing import Dict
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from sequencing_data import SequencingData
from utils.seq_data_utils import heatmap_masks


def build_gaussian_model_2d(df_x: pd.DataFrame, df_y: pd.DataFrame) -> GaussianMixture:
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
    width : float
        Width of the ellipse
    height: float
        Height of the ellipse
    angle : float
        Angle of the ellipse
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


def get_ellipses(
    gaussian_model: GaussianMixture, sigma_cutoff: int
) -> Dict[int, list[float, float, float, float]]:
    """
    Calculate the ellipses to be drawn for each sigma value up to sigma_cutoff using a
    2-D Gaussian model

    Parameters
    ----------
    gaussian_model : GaussianMixture
        2D Gaussian model
    sigma_cutoff : int
        Number of sigma to calculate ellipses for

    Returns
    -------
    ellipses : dict
        Dictionary of sigma values and the corresponding ellipses.
    """
    ellipses = {}
    for center, covar in zip(gaussian_model.means_, gaussian_model.covariances_):
        # a loop here in case there's more than one for some reason
        width, height, angle = ellipse_coordinates(covar)
    for n_sigma in range(1, sigma_cutoff + 1):
        width_sigma = n_sigma * width
        height_sigma = n_sigma * height
        ellipses[n_sigma] = [
            center,
            width_sigma,
            height_sigma,
            angle,
        ]
    return ellipses


def determine_significance_2d(
    df_x: pd.DataFrame, df_y: pd.DataFrame, ellipse: list[float, float, float, float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Takes two sets of data and an ellipse and determines the points that lie outside
    the bounds of significance

    Parameters
    ----------
    df_x : pd.DataFrame
        First dataset
    df_y : pd.DataFrame
        Second dataset
    ellipse : list[float, float, float, float]
        Center, width, height, and angle of ellipse

    Returns
    -------
    df_significant_sensitive : pd.DataFrame
        Dataframe of boolean values for which positions have significant values for drug
        sensitivity according to 2D Gaussian model
    df_significant_sensitive : pd.DataFrame
        Dataframe of boolean values for which positions have significant values for drug
        resistance according to 2D Gaussian model
    """
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

    def is_significant_sensitive(point):
        return is_outside_ellipse(point) and is_negative_quadrant(point)

    def is_significant_resistant(point):
        return is_outside_ellipse(point) and is_positive_quadrant(point)

    # * merge two dataframes element-wise for significance test
    df_xy = pd.concat([df_x, df_y]).groupby(level=0, axis=0).agg(list)

    df_significant_sensitive = df_xy.applymap(is_significant_sensitive)
    df_significant_resistant = df_xy.applymap(is_significant_resistant)

    return df_significant_sensitive, df_significant_resistant


def gaussian_significance_2d(
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    sigma_cutoff: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[int, list[float, float, float, float]]]:
    """
    Builds 2D Gaussian model, calculates ellipses for each sigma, and determines the
    identity of the datapoints that lie outside of the outer ellipse

    Parameters
    ----------
    df_x : pd.DataFrame
        First dataframe
    df_y : pd.DataFrame
        Second dataframe
    sigma_cutoff : int, optional
        Number of sigma to draw ellipses for, by default 4

    Returns
    -------
    df_significant_sensitive : pd.DataFrame
        Dataframe of boolean values for which positions have significant values for drug
        sensitivity according to 2D Gaussian model
    df_significant_sensitive : pd.DataFrame
        Dataframe of boolean values for which positions have significant values for drug
        resistance according to 2D Gaussian model
    ellipses : Dict[int, tuple[float, float, float, float]]
        Ellipses calculated for determination of significance and drawing for plotting
    """
    gaussian_model = build_gaussian_model_2d(df_x, df_y)
    ellipses = get_ellipses(gaussian_model, sigma_cutoff=sigma_cutoff)
    significance_ellipse = ellipses[sigma_cutoff]
    df_significant_sensitive, df_significant_resistant = determine_significance_2d(
        df_x, df_y, significance_ellipse
    )
    return df_significant_sensitive, df_significant_resistant, ellipses


def significance_sigma_dfs_2d(
    data: SequencingData,
    read_threshold: int = 20,
    sigma_cutoff: int = 4,
) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Extracts the significant residue positions from the fitness dataframes according
    to the 2D Gaussian model for significance

    Parameters
    ----------
    data : SequencingData
        Data from experiment sequencing with count-, enrichment-, and fitness-values
    read_threshold : int, optional
        Minimum number of reads required to be included, by default 20
    sigma_cutoff : int, optional
        How many sigmas away from the synonymous mutation values to use as the
        cutoff for significance, by default 4

    Returns
    -------
    significant_sensitive_dfs : Dict[str, pd.DataFrame]
        Set of dataframes with boolean values indicating which mutations are relevant for
        drug sensitivity
    significant_resistant_dfs : Dict[str, pd.DataFrame]
        Set of dataframes with boolean values indicating which mutations are relevant for
        drug sensitivity
    """
    gene = data.gene
    fitness_dict = data.fitness
    wt_mask = heatmap_masks(gene)
    significant_sensitive_dfs = {}
    significant_resistant_dfs = {}
    dfs_filtered = data.filter_fitness_read_noise(read_threshold=read_threshold)
    drugs = set(re.sub("_[0-9]$", "", x) for x in fitness_dict)
    for drug in drugs:
        x, y = data.get_pairs(drug, data.samples)
        df_x = dfs_filtered[x].mask(wt_mask)
        df_y = dfs_filtered[y].mask(wt_mask)

        significant_sensitive, significant_resistant, _ = gaussian_significance_2d(
            df_x,
            df_y,
            sigma_cutoff=sigma_cutoff,
        )
        significant_sensitive_dfs[drug] = significant_sensitive
        significant_resistant_dfs[drug] = significant_resistant
    return significant_sensitive_dfs, significant_resistant_dfs


def significance_sigma_mutations_2d(
    data: SequencingData,
    read_threshold: int = 20,
    sigma_cutoff: int = 4,
) -> pd.DataFrame:
    """
    For replica measurements, build a 2D Gaussian model and generate a dataframe of the
    list of mutations that are determined to have significant fitness values by the model

    Parameters
    ----------
    data : SequencingData
        Data from experiment sequencing with count-, enrichment-, and fitness-values
    read_threshold : int, optional
        Minimum number of reads for fitness value to be considered valid, by default 20
    sigma_cutoff : int, optional
        Number of sigma to draw ellipses for, by default 4

    Returns
    -------
    df_all_fitness_sigma : pd.DataFrame
        Table for significance of all mutations according to the 2D Gaussian model
    """
    gene = data.gene
    fitness_dict = data.fitness
    cds_translation = gene.cds_translation
    drugs_all = sorted(drug for drug in data.treatments if "UT" not in drug)
    significant_sensitive_dfs, significant_resistant_dfs = significance_sigma_dfs_2d(
        data,
        read_threshold=read_threshold,
        sigma_cutoff=sigma_cutoff,
    )

    sample_name = list(fitness_dict)[0]
    # * get a list of all possible position-residue mutations
    point_mutations = pd.MultiIndex.from_product(
        [fitness_dict[sample_name].index, fitness_dict[sample_name].columns]
    ).values
    list_all_fitness = []

    for drug in drugs_all:
        replica_one, replica_two = data.get_pairs(drug, fitness_dict)
        df1 = data.filter_fitness_read_noise(read_threshold)[replica_one]
        df2 = data.filter_fitness_read_noise(read_threshold)[replica_two]

        significant_sensitive = significant_sensitive_dfs[drug]
        significant_resistant = significant_resistant_dfs[drug]

        for position, residue in list(point_mutations):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                fitness_entry = {
                    "aa_pos": position + 1,
                    "ref_aa": cds_translation[position],
                    "query_aa": residue,
                    "drug": drug,
                    "rel_fitness_1": df1.loc[position, residue],
                    "rel_fitness_2": df2.loc[position, residue],
                    "rel_fitness_mean": np.nanmean(
                        [df1.loc[position, residue], df2.loc[position, residue]]
                    ),
                    "significant": (
                        significant_sensitive.loc[position, residue]
                        | significant_resistant.loc[position, residue]
                    ),
                }
            if fitness_entry["significant"]:
                if significant_sensitive.loc[position, residue]:
                    fitness_entry.update({"type": "sensitive"})
                elif significant_resistant.loc[position, residue]:
                    fitness_entry.update({"type": "resistance"})
            list_all_fitness.append(fitness_entry)

    df_all_fitness_sigma = pd.DataFrame(list_all_fitness)
    rel_fitness_1 = df_all_fitness_sigma["rel_fitness_1"]
    rel_fitness_2 = df_all_fitness_sigma["rel_fitness_2"]
    # make sure both replicates meet noise threshold requirements
    # drop any position where one value is NaN
    below_read_threshold_index = (
        rel_fitness_1.where(np.isfinite(rel_fitness_1) & np.isfinite(rel_fitness_2))
        .dropna()
        .index
    )
    df_all_fitness_sigma = df_all_fitness_sigma.loc[below_read_threshold_index]
    return df_all_fitness_sigma


### 1-D Gaussian modeling


def build_gaussian_model_1d(df: pd.DataFrame) -> tuple[float, float]:
    """
    Use relative fitness values of synonymous mutations to generate a 1-D Gaussian model
    that will determine the bounds of significance for fitness effects

    Parameters
    ----------
    df : pd.DataFrame
        Relative fitness data for drug

    Returns
    -------
    mu : float
        Mean for 1D Gaussian model
    std : float
        Standard deviation for 1-D Gaussian model
    """

    # * 1-D gaussian model fitting for all mutations
    x = df
    X = x.values.flatten()
    X = X[~np.isnan(X)]
    # * 1-D gaussian model fitting for synonymous mutations
    x_syn = x["∅"]
    X_syn = x_syn.values.flatten()
    X_syn = X_syn[~np.isnan(X_syn)]
    # * fit 1-D gaussian model
    mu, std = norm.fit(X_syn)

    return mu, std


def gaussian_significance_1d(
    df: pd.DataFrame, sigma_cutoff: int = 4, use_synonymous: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Uses mean and standard deviation to determine the 1D Gaussian model and determine
    the significance of the relative fitness values for each mutation

    Parameters
    ----------
    df : pd.DataFrame
        Relative fitness data for drug
    sigma_cutoff : int, optional
        Number of sigma to use to calculate significance, by default 4
    use_synonymous : bool, optional
        Whether to build a 1-D model using just the synonymous mutations or not, by default True

    Returns
    -------
    df_significant_sensitive : pd.DataFrame
        Dataframe of boolean values for which positions have significant values for drug
        sensitivity according to 1D Gaussian model
    df_significant_resistant : pd.DataFrame
        Dataframe of boolean values for which positions have significant values for drug
        resistance according to 1D Gaussian model
    """

    if use_synonymous:
        mu, std = build_gaussian_model_1d(df)
    else:
        mu, std = build_gaussian_model_1d_mixture(df)

    # determine significance in 1D
    def is_significant(value):
        if np.isnan(value).any():
            return False
        x = value
        return x < (mu - sigma_cutoff * std) or (x > (mu + sigma_cutoff * std))

    def is_positive_quadrant(value):
        return (np.array(value) >= 0).all()

    def is_negative_quadrant(value):
        return (np.array(value) <= 0).all()

    def is_significant_sensitive(value):
        return is_significant(value) and is_negative_quadrant(value)

    def is_significant_resistant(value):
        return is_significant(value) and is_positive_quadrant(value)

    df_significant_sensitive = df.applymap(is_significant_sensitive)
    df_significant_resistant = df.applymap(is_significant_resistant)

    return df_significant_sensitive, df_significant_resistant


def significance_sigma_dfs_1d(
    data: SequencingData,
    read_threshold: int = 20,
    sigma_cutoff: int = 4,
    use_synonymous: bool = True,
) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Extracts the significant residue positions from the fitness dataframes according
    to the 1D Gaussian model for significance

    Parameters
    ----------
    data : SequencingData
        Data from experiment sequencing with count-, enrichment-, and fitness-values
    read_threshold : int, optional
        Minimum number of reads required to be included, by default 20
    sigma_cutoff : int, optional
        How many sigmas away from the synonymous mutation values to use as the
        cutoff for significance, by default 4
    use_synonymous : bool, optional
        Whether to build a 1-D model using just the synonymous mutations or not, by default True

    Returns
    -------
    significant_sensitive_dfs : Dict[str, pd.DataFrame]
        Set of dataframes with boolean values indicating which mutations are relevant for
        drug sensitivity
    significant_resistant_dfs : Dict[str, pd.DataFrame]
        Set of dataframes with boolean values indicating which mutations are relevant for
        drug sensitivity
    """
    gene = data.gene
    fitness_dict = data.fitness
    wt_mask = heatmap_masks(gene)
    significant_sensitive_dfs = {}
    significant_resistant_dfs = {}
    dfs_filtered = data.filter_fitness_read_noise(read_threshold=read_threshold)
    drugs = set(re.sub("_[0-9]$", "", x) for x in fitness_dict)
    for drug in drugs:
        df = dfs_filtered[drug].mask(wt_mask)

        significant_sensitive, significant_resistant = gaussian_significance_1d(
            df, sigma_cutoff=sigma_cutoff, use_synonymous=use_synonymous
        )
        significant_sensitive_dfs[drug] = significant_sensitive
        significant_resistant_dfs[drug] = significant_resistant
    return significant_sensitive_dfs, significant_resistant_dfs


def significance_sigma_mutations_1d(
    data: SequencingData,
    read_threshold: int = 20,
    sigma_cutoff: int = 4,
    use_synonymous: bool = True,
) -> pd.DataFrame:
    """
    For pooled measurements, build a 1D Gaussian model and generate a dataframe of the
    list of mutations that are determined to have significant fitness values by the model

    Parameters
    ----------
    data : SequencingData
        Data from experiment sequencing with count-, enrichment-, and fitness-values
    read_threshold : int, optional
        Minimum number of reads for fitness value to be considered valid, by default 20
    sigma_cutoff : int, optional
        Number of sigma to draw ellipses for, by default 4
    use_synonymous : bool, optional
        Whether to build a 1-D model using just the synonymous mutations or not, by default True

    Returns
    -------
    df_all_fitness_sigma : pd.DataFrame
        Table for significance of all mutations according to the 1D Gaussian model
    """
    gene = data.gene
    fitness_dict = data.fitness
    cds_translation = gene.cds_translation
    drugs_all = sorted(drug for drug in data.treatments if "UT" not in drug)
    significant_sensitive_dfs, significant_resistant_dfs = significance_sigma_dfs_1d(
        data,
        read_threshold=read_threshold,
        sigma_cutoff=sigma_cutoff,
        use_synonymous=use_synonymous,
    )

    sample_name = list(fitness_dict)[0]
    # * get a list of all possible position-residue mutations
    point_mutations = pd.MultiIndex.from_product(
        [fitness_dict[sample_name].index, fitness_dict[sample_name].columns]
    ).values
    list_all_fitness = []

    for drug in drugs_all:
        df = fitness_dict[drug]

        significant_sensitive = significant_sensitive_dfs[drug]
        significant_resistant = significant_resistant_dfs[drug]

        for position, residue in list(point_mutations):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                fitness_entry = {
                    "aa_pos": position + 1,  # * correct index to start with '1',
                    "ref_aa": cds_translation[position],
                    "query_aa": residue,
                    "drug": drug,
                    "rel_fitness": df.loc[position, residue],
                    "significant": (
                        significant_sensitive.loc[position, residue]
                        | significant_resistant.loc[position, residue]
                    ),
                }
                if significant_sensitive.loc[position, residue]:
                    fitness_entry.update({"type": "sensitive"})
                elif significant_resistant.loc[position, residue]:
                    fitness_entry.update({"type": "resistance"})
                list_all_fitness.append(fitness_entry)

    df_all_fitness_sigma = pd.DataFrame(list_all_fitness)
    return df_all_fitness_sigma


def build_gaussian_model_1d_mixture(
    df: pd.DataFrame, n_components: int = 3
) -> tuple[float, float]:
    """
    Use relative fitness values of all mutations to build a Gaussian mixture model
    that will determine the 1-D bounds of significance for fitness effects

    Parameters
    ----------
    df : pd.DataFrame
        Relative fitness data for drug
    n_components : int, optional
        Number of components to use for Gaussian mixture model

    Returns
    -------
    mu : float
        Mean for 1-D Gaussian model
    std : float
        Standard deviation for 1-D Gaussian model
    """

    # * 1-D gaussian model fitting for all mutations
    x = df
    X = x.values.flatten()
    X = X[~np.isnan(X)]
    X = np.expand_dims(X, 1)

    # * build and fit Gaussian mixture model
    model = GaussianMixture(n_components=n_components, covariance_type="full")
    model.fit(X)
    means = model.means_
    covars = model.covariances_

    # * determine cutoffs significance
    # calculate standard deviation from covariances
    stdevs = np.sqrt(covars)

    # * central peak
    central_mu_idx = np.abs(means).argmin()
    mu = means[central_mu_idx][0]
    std = stdevs[central_mu_idx][0][0]

    return mu, std
