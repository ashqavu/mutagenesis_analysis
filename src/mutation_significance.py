#!/usr/bin/env python
"""
This script uses Gaussian modeling to determine which mutations are significant and visualize them in scatterplots
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

from visualization import filter_fitness_read_noise


def ellipse_coordinates(covariance):
    """
    Given a position and covariance, calculate the boundaries of the ellipse to be drawn
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

def ellipses_draw(center, width, height, angle, sigma_cutoff=3, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    for n_sigma in range(1, sigma_cutoff+1):
        ax.add_patch(Ellipse(center, n_sigma * width, n_sigma * height, angle=angle, ec="k", lw=0.667, fill=None, **kwargs))
    return (sigma_cutoff * width) / 2, (sigma_cutoff * height) / 2, np.radians(angle)

def gaussian_significance_model(treatment, counts_dict, fitness_dict, gene, ax=None, read_threshold=1, sigma_cutoff=3, xlim=(-2.5, 2.5), ylim=(-2.5, 2.5)):
    if ax is None:
        ax = plt.gca()
    # * pick pairs for each drug
    treatment_pair = [key for key in fitness_dict.keys() if treatment in key]
    if not treatment_pair:
        raise KeyError(f"No fitness data: {treatment}")
    if len(treatment_pair) > 2:
        raise IndexError("Treatment has more than 2 replicates to compare")

    replica_one = treatment_pair[0]
    replica_two = treatment_pair[1]
    df_x = filter_fitness_read_noise(replica_one, counts_dict, fitness_dict, gene, read_threshold)
    df_y = filter_fitness_read_noise(replica_two, counts_dict, fitness_dict, gene, read_threshold)

    # * merge two dataframes element-wise for significance test
    df_xy = pd.concat([df_x, df_y]).groupby(level=0, axis=0).agg(list)

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

    # * build numpy matrix for gaussian model plotting
    X = np.column_stack((df_x.values.flatten(), df_y.values.flatten()))
    # filter NaN in pairs
    X = X[~np.isnan(X).any(axis=1)]

    # * drawing ellipses given position and covariance
    for center, covar in zip(gaussian_model.means_, gaussian_model.covariances_):
        # calculate ellipse paramaters
        width, height, angle = ellipse_coordinates(covar)
        # draw ellipses
        width_f, height_f, angle_f = ellipses_draw(
            center, width, height, angle, sigma_cutoff=sigma_cutoff, ax=ax, zorder=10
        )
    # * determine significance
    def is_outside_ellipse(point):
        # if either fitness value is NaN, discard point
        if np.isnan(point).any():
            return False
        x, y = point
        cos_angle = np.cos(angle_f)
        sin_angle = np.sin(angle_f)
        x_center = center[0]
        y_center = center[1]
        x_dist = x - x_center
        y_dist = y - y_center
        a = ((x_dist * cos_angle) + (y_dist * sin_angle)) ** 2
        b = ((x_dist * sin_angle) - (y_dist * cos_angle)) ** 2
        limit = (width_f**2) * (height_f**2)
        return a * (height_f**2) + b * (width_f**2) > limit

    def is_positive_quadrant(point):
        return (np.array(point) >= 0).all()

    def is_negative_quadrant(point):
        return (np.array(point) <= 0).all()

    def is_sign_resistant(point):
        return is_outside_ellipse(point) and is_positive_quadrant(point)

    def is_sign_sensitive(point):
        return is_outside_ellipse(point) and is_negative_quadrant(point)

    sign_resistant = df_xy.applymap(is_sign_resistant)
    sign_sensitive = df_xy.applymap(is_sign_sensitive)

    # * plotting
    # all mutations
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        zorder=-1,
        ax=ax,
        plotnonfinite=False,
        color="gray",
        lw=2,
        s=10,
    )
    # synonymous mutations
    sns.scatterplot(
        x=df_x["∅"],
        y=df_y["∅"],
        ax=ax,
        plotnonfinite=False,
        color="yellowgreen",
        lw=0.5,
        s=10,
    )
    # resistant mutations
    sns.scatterplot(
        x=df_x[sign_resistant].values.flatten(),
        y=df_y[sign_resistant].values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="lightcoral",
        lw=0.5,
        s=10,
    )
    # sensitive mutations
    sns.scatterplot(
        x=df_x[sign_sensitive].values.flatten(),
        y=df_y[sign_sensitive].values.flatten(),
        ax=ax,
        plotnonfinite=False,
        color="dodgerblue",
        lw=0.5,
        s=10,
    )

    # * axis lines and limits
    # diagonal
    ax.plot([-4, 4], [-4, 4], ":", color="gray", alpha=0.5, zorder=0)
    # x-axis
    ax.plot([0, 0], [-4, 4], "-", color="gray", alpha=0.5, lw=1, zorder=0)
    #y y-axis
    ax.plot([-4, 4], [0, 0], "-", color="gray", alpha=0.5, lw=1, zorder=0)
    ax.set(xlim=xlim, ylim=ylim, anchor="NW", aspect="equal")
    ax.set_xlabel(replica_one, fontweight="bold")
    ax.set_ylabel(replica_two, fontweight="bold")

    return sign_resistant, sign_sensitive