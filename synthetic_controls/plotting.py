import matplotlib.pyplot as plt

from synthetic_controls.estimators import BaseEstimator


def plot_outcomes(
    estimator: BaseEstimator,
    observed_outcomes=True,
    synthetic_outcomes=True,
    ax=None,
    color_outcome="C0",
    color_synthetic="C1",
    legend=True,
    show_t0=True,
    synthetic_name=None,
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
    color_outcome : str (default: "C0")
        The color of the observed outcomes line.
    color_synthetic : str (default: "C1")
        The color of the synthetic outcomes line.
    legend : bool (default: True)
        Whether to show the legend.
    show_t0 : bool (default: True)
        Whether to show a vertical line at the treatment start time.
    synthetic_name : str (optional)
        The name of the synthetic outcomes. If None, `"Synthetic" + target name` is used.

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
        ax.plot(
            x_axis,
            observed_outcomes,
            label="Observed " + estimator.target_name,
            color=color_outcome,
        )

    if synthetic_outcomes:
        synthetic_outcomes = estimator.synthetic_outcomes
        label = (
            synthetic_name
            if synthetic_name is not None
            else "Synthetic " + estimator.target_name
        )
        l = ax.plot(
            x_axis,
            synthetic_outcomes,
            label=label,
            linestyle="--",
            color=color_synthetic,
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

    if show_t0:
        ax.axvline(
            estimator.treatment_start_time,
            color="black",
            linestyle="--",
        )

    ax.set_xlabel(estimator.dataframe.index.name)
    ax.set_ylabel("Outcome")
    if legend:
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
    if len(estimators) == 0:
        raise ValueError("No estimators to plot.")
    if ax is None:
        fig, ax = plt.subplots()

    plot_outcomes(
        estimators[0],
        ax=ax,
        observed_outcomes=True,
        synthetic_outcomes=True,
        legend=False,
        synthetic_name=estimators[0].__class__.__name__,
    )
    for i, estimator in enumerate(estimators[1:]):
        plot_outcomes(
            estimator,
            ax=ax,
            observed_outcomes=False,
            synthetic_outcomes=True,
            color_synthetic=f"C{i+2}",
            legend=False,
            synthetic_name=estimator.__class__.__name__,
        )

    ax.legend()
    return ax
