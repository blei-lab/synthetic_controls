import copy
from typing import List, Iterable
from typing import Optional, Union

import numpy as np
import pandas as pd


class ObservedCausesData:
    """
    The observed causes data for the M-and-James-bond estimators.
    Contains the distributions of the causes and the Lipschitz constants.

    Parameters
    ----------
    distributions_of_causes
        A dictionary of distributions of the causes, where the keys are the units and the
        values are Distribution objects.
    lipschitz_constants
        A dictionary of Lipschitz constants for the causes. The keys are the names of the
        dimensions of the causes and the values are the Lipschitz constants for each
        dimension. If dimensions have different Lipschitz constants, they will be scaled
        to have the same Lipschitz constant.
    """
    def __init__(
        self, distributions_of_causes: dict, lipschitz_constants: Optional[dict] = None
    ):
        self._distributions_of_causes = distributions_of_causes
        if not self._distributions_of_causes:
            raise ValueError("The dictionary of distributions of causes is empty.")
        Distribution.homogenize_atoms(self._distributions_of_causes.values())
        self.dim_names_causes = next(
            iter(self._distributions_of_causes.values())
        ).dim_names
        self.n_dim_causes = len(self.dim_names_causes)
        self._lipschitz_constants = lipschitz_constants

        self._optimal_distributions_of_causes = None
        self._optimal_lipschitz_constants = None

    def get_optimal_distributions_of_causes_and_lipschitz_constant(self):
        if self._lipschitz_constants is None:
            raise ValueError("The Lipschitz constants are not set.")
        optimal_distributions_of_causes = copy.deepcopy(self._distributions_of_causes)
        normalizer = np.array(
            [self._lipschitz_constants[dim] for dim in self.dim_names_causes]
        )
        optimal_lipschitz_constants = {dim: 1.0 for dim in self.dim_names_causes}
        for unit, distribution in optimal_distributions_of_causes.items():
            distribution.scale_atoms(normalizer)

        self._optimal_distributions_of_causes = optimal_distributions_of_causes
        self._optimal_lipschitz_constants = optimal_lipschitz_constants

        return optimal_distributions_of_causes, 1

    @staticmethod
    def from_dataframe_probabilities(
        dataframe: pd.DataFrame, unit_id_col: str, probability_col: str
    ):
        distributions_of_causes = {
            unit: Distribution.from_dataframe_probability(
                dataframe[dataframe[unit_id_col] == unit].drop(columns=unit_id_col),
                probability_column=probability_col,
            )
            for unit in dataframe[unit_id_col].unique()
        }
        return ObservedCausesData(distributions_of_causes)

    def set_lipschitz(self, lipschitz_constants: Union[dict, List, float]):
        if isinstance(lipschitz_constants, float) and self.n_dim_causes:
            lipschitz_constants = {
                n: lipschitz_constants for n in self.dim_names_causes
            }
        if isinstance(lipschitz_constants, list):
            if len(lipschitz_constants) != self.n_dim_causes:
                raise ValueError(
                    "The list of Lipschitz constants must have the same length "
                    "as the number of dimensions of the causes."
                )
            lipschitz_constants = {
                n: lipschitz_constants[i] for i, n in enumerate(self.dim_names_causes)
            }
        # check that the keys of the dictionary are the same as the causes names
        if set(lipschitz_constants.keys()) != set(self.dim_names_causes):
            raise ValueError(
                "The keys of the dictionary of Lipschitz constants must be the "
                "same as the names of the dimensions of the causes."
            )
        self._lipschitz_constants = lipschitz_constants


class Distribution:
    """
    A distribution over a finite set of atoms.
    """
    def __init__(
        self, atoms: np.ndarray, probabilities: np.ndarray, dim_names: List[str] = None
    ):
        self.atoms = np.array(atoms)
        self.probabilities = np.array(probabilities)
        self.n_dim = self.atoms.shape[1]
        self.dim_names = dim_names or [f"dim_{i}" for i in range(self.atoms.shape[1])]

    def __str__(self):
        return f"Distribution with {len(self.atoms)} atoms"

    def __repr__(self):
        return str(self)

    def scale_atoms(self, scale_factor):
        """
        Normalize the atoms in-place using the given normalizers.
        """
        self.atoms = self.atoms * scale_factor

    @staticmethod
    def homogenize_atoms(distributions: Iterable["Distribution"]):
        """
        Ensure that all distributions in the iterable have the same atoms,
        by adding missing atoms with probability 0 (in-place).
        """
        all_atoms = set()
        for d in distributions:
            for a in d.atoms:
                all_atoms.add(tuple(a))
        all_atoms = list(all_atoms)
        atoms_to_id = dict([(a, i) for i, a in enumerate(all_atoms)])
        all_atoms = np.array([list(a) for a in all_atoms])
        for d in distributions:
            probabilities = np.zeros(len(all_atoms))
            for i, a in enumerate(d.atoms):
                probabilities[atoms_to_id[tuple(a)]] = d.probabilities[i]
            d.atoms = all_atoms.copy()
            d.probabilities = probabilities

    @staticmethod
    def from_dataframe_probability(data, probability_column="probability"):
        probabilities = data[probability_column].values.astype(float)
        atoms = data.drop(columns=[probability_column])
        return Distribution(atoms.values, probabilities, atoms.columns.tolist())
