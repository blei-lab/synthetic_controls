import functools

import numpy as np
import pandas as pd

from synthetic_controls.observed_causes_data import ObservedCausesData


def set_cwd_to_this_file(func):
    """Decorator to set cwd to this file's directory.

    Restores the original cwd after the function is called, or if an exception is raised.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import os

        old_cwd = os.getcwd()
        os.chdir(os.path.dirname(__file__))
        try:
            return func(*args, **kwargs)
        finally:
            os.chdir(old_cwd)

    return wrapper


@set_cwd_to_this_file
def simulated_age_groups(seed=0):
    panel_data = pd.read_csv(
        "./datasets/synthetic_age_groups_outcomes.csv", index_col=0
    )
    rng = np.random.default_rng(seed)
    panel_data += rng.normal(0, 1, panel_data.shape)

    synthetic_distributions = pd.read_csv(
        "./datasets/synthetic_age_groups_distributions.csv", index_col=0
    )
    observed_causes_data = ObservedCausesData.from_dataframe_probabilities(
        synthetic_distributions, "geo", "prob"
    )
    observed_causes_data.set_lipschitz(4.0)

    return panel_data, observed_causes_data


@set_cwd_to_this_file
def tobacco_us_states():
    panel_data = pd.read_csv(
        "./datasets/us_states_cigarette_consumption_per_capita.csv", index_col=0
    )
    # Column is duplicated with `District of Columbia` and `District Of Columbia`
    panel_data.drop(columns=["District Of Columbia"], inplace=True)

    census_distribution = pd.read_csv(
        "./datasets/us_states_census_race_sex_age.csv", index_col=0
    )
    observed_causes_data = ObservedCausesData.from_dataframe_probabilities(
        census_distribution, "geo", "prob"
    )

    # Lipschitz constant per cause: estimated from survey data in number of cigarettes per day
    # Categorical variables are encoded as one-hot vectors: todo explain
    lipschitz_per_cause = {
        "age": 0.27,
        "sex_male": 2.77 / 2,
        "sex_female": 2.77 / 2,
        "race_asian": 2.442203,
        "race_black": 1.480483,
        "race_mix": 2.507990,
        "race_native_american": 0.372803,
        "race_native_pacific": 0.137936,
        "race_other": 0,
        "race_white_hispanic": 2.378094,
        "race_white_non_hispanic": 3.557923,
    }
    # Convert to number of cigarettes packs per year (the unit of the outcomes)
    lipschitz_per_cause = {k: v * 365 / 20 for k, v in lipschitz_per_cause.items()}
    observed_causes_data.set_lipschitz(lipschitz_per_cause)

    return panel_data, observed_causes_data
