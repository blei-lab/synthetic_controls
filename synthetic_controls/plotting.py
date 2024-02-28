import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from synthetic_controls._estimator import BaseEstimator


def plot_outcomes(
    estimator: BaseEstimator, observed_outcomes=True, synthetic_outcomes=True, ax=None
):
    """Plot the outcomes of the estimator.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to plot the outcomes of.
    observed_outcomes : bool, default True
        Whether to plot the observed outcomes.
    synthetic_outcomes : bool, default True
        Whether to plot the synthetic outcomes.
    ax : matplotlib.pyplot.Axes, optional
        The axes to plot on. If None, a new figure and axes are created.

    Returns
    -------
    ax : matplotlib.pyplot.Axes
        The axes the outcomes were plotted on.
    """
    if estimator.target_name is None:
        raise ValueError("The estimator is not fitted yet.")

    if ax is None:
        fig, ax = plt.subplots()

    x_axis = estimator.dataframe.index
    if observed_outcomes:
        observed_outcomes = estimator.target_outcomes
        ax.plot(x_axis, observed_outcomes, label="Observed " + estimator.target_name)

    if synthetic_outcomes:
        synthetic_outcomes = estimator.synthetic_outcomes
        l = ax.plot(
            x_axis,
            synthetic_outcomes,
            label="Synthetic " + estimator.target_name,
            linestyle="--",
        )
        if hasattr(estimator, "get_bound"):
            bound = estimator.get_bound()
            ax.fill_between(
                x_axis,
                synthetic_outcomes - bound,
                synthetic_outcomes + bound,
                alpha=0.3,
                color=l[0].get_color(),
            )

    ax.axvline(
        estimator.treatment_start_time, color="black", linestyle="--", label="T0"
    )

    ax.set_xlabel(estimator.dataframe.index.name)
    ax.set_ylabel("Outcome")
    ax.legend()

    return ax


def compare_estimators(estimators, ax=None):
    """Plot the outcomes of multiple estimators.

    Parameters
    ----------
    estimators : list of BaseEstimator
        The estimators to plot the outcomes of.
    ax : matplotlib.pyplot.Axes, optional
        The axes to plot on. If None, a new figure and axes are created.

    Returns
    -------
    ax : matplotlib.pyplot.Axes
        The axes the outcomes were plotted on.
    """
    raise NotImplementedError("This function is not implemented yet.")
    if ax is None:
        fig, ax = plt.subplots()

    for estimator in estimators:
        plot_outcomes(
            estimator,
            observed_outcomes=observed_outcomes,
            synthetic_outcomes=synthetic_outcomes,
            ax=ax,
        )

    return ax
