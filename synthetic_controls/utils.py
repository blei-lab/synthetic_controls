from typing import List, Iterable

import numpy as np


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

    # @staticmethod
    # def from_df_samples(data):
