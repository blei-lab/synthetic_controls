import abc
import cvxpy as cp
import numpy as np

from synthetic_controls.observed_causes_data import ObservedCausesData, Distribution


class BaseEstimator(abc.ABC):
    def __init__(self, dataframe, *, treatment_start_time=None, **kwargs):
        """Initialize the synthetic control estimator.

        Parameters
        ----------
        treatment_start_time : int, optional
            Time at which the treatment starts, such that `X[:treatment_start_time]`
            is the untreated data and `X[treatment_start_time:]` is the treated data.

        """
        self.dataframe = dataframe
        self.treatment_start_time = treatment_start_time
        self._weights = None

        self.donors_outcomes = None
        self.target_outcomes = None
        self.donor_names = None
        self.target_name = None

        self._n_donors = None
        self._objective_function = None
        self._objective = None
        self._constraints = None
        self._problem = None

    @property
    def treatment_start_time(self):
        return self._treatment_start_time

    @treatment_start_time.setter
    def treatment_start_time(self, value):
        if value is None:
            self._treatment_start_time = None
            return
        if value not in self.dataframe.index:
            raise ValueError(
                "The treatment start time should be one of the index values"
            )
        self._treatment_start_time = value
        index = self.dataframe.index.to_list().index(value)
        self._treatment_start_time_index = index

    @property
    def weights(self):
        if self._weights is None:
            raise ValueError("Weights have not been set yet.")
        return dict(zip(self.donor_names, self._weights.value))

    def fit(
        self, target_name, donor_names=None, treatment_start_time=None, verbose=False
    ):
        """Fit the synthetic control estimator to the data.

        Parameters
        ----------
        target_name : str
            Name of the target unit in the panel data.
        donor_names : list of str, optional
            Names of the donor units in the panel data.
            If `None`, it is set to all columns of `self.dataframe` except `target_name`.
        treatment_start_time : int, optional
            Time at which the treatment starts, such that `X[:treatment_start_time]`
            is the untreated data and `X[treatment_start_time:]` is the treated data.
        verbose : bool, optional
            Set the verbosity level of the optimization solver.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if treatment_start_time is not None:
            self.treatment_start_time = treatment_start_time
        elif self.treatment_start_time is None:
            raise ValueError(
                "The treatment start time must be set either in the constructor"
                " or in the `fit` method."
            )

        if donor_names is None:
            donor_names = self.dataframe.columns.drop(target_name)
        self.target_name = target_name
        self.donor_names = donor_names
        self._n_donors = len(donor_names)

        self.donors_outcomes = self.dataframe[donor_names].values
        self.target_outcomes = self.dataframe[target_name].values
        self._fit(verbose=verbose)
        return self

    @property
    def _synthetic_outcomes(self):
        return self.donors_outcomes @ self._weights

    @property
    def synthetic_outcomes(self):
        return self._synthetic_outcomes.value

    def plot_outcomes(self, observed_outcomes=True, synthetic_outcomes=True, ax=None):
        """Plot the outcomes of the target and synthetic units."""
        from plotting import plot_outcomes

        plot_outcomes(
            self,
            observed_outcomes=observed_outcomes,
            synthetic_outcomes=synthetic_outcomes,
            ax=ax,
        )

    def _fit(self, verbose=False):
        # Set up the optimization problem.
        self._weights = cp.Variable(self._n_donors)

        # Form the optimization problem.
        self._objective_function = self._get_objective_function()
        self._objective = cp.Minimize(self._objective_function)
        self._constraints = self._get_constraints()
        self._problem = cp.Problem(self._objective, self._constraints)

        # Solve the optimization problem.
        self._problem.solve(solver=cp.CLARABEL, verbose=verbose)

    def predict(self, X=None):
        """Predict the counterfactual outcome of the target unit.

        Parameters
        ----------
        X : array-like, shape (time, donors) or None
            If array-like, it is the panel data of the donor units.
            If None, it is the same as the panel data used to fit the estimator.

        Returns
        -------
        y_pred : array-like, shape (time, )
            The counterfactual outcome of the target unit.
        """
        if X is None:
            X = self.donors_outcomes
        return X @ self.weights

    def get_weight_summary(self, top_k=None, threshold=1e-5):
        """
        Get a summary of the estimated weights of the donors.
        Only donors with a weight above the threshold are shown.

        Parameters
        ----------
        top_k : int, optional
            Number of top donors to show in the summary. If None, all the donors
            meeting the threshold are shown.
        threshold : float, optional
            Threshold for the weights of the donors to be shown in the summary.

        Returns
        -------
        Dict[str, float]
            Dictionary of donor names and their estimated weights.
        """
        return dict(
            filter(
                lambda x: x[1] > threshold,
                sorted(
                    self.weights.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:top_k],
            )
        )

    def _get_constraints_weights_simplex(self):
        return [cp.sum(self._weights) == 1, self._weights >= 0]

    def _get_objective_function_l2_pretreatment(self):
        """The L2 norm between the synthetic control and the treated unit
        outcomes over the pre-treatment period."""
        error = self._synthetic_outcomes - self.target_outcomes
        error = error[: self._treatment_start_time_index]
        return cp.sum_squares(error)

    @abc.abstractmethod
    def _get_objective_function(self):
        """Get the objective function for the optimization problem.

        E.g., the L2 norm between the synthetic control and the treated unit.

        Returns
        -------
        cvxpy.expression
            The objective function for the optimization problem.
        """
        pass

    @abc.abstractmethod
    def _get_constraints(self):
        """Get the constraints for the optimization problem.

        E.g., the weights must be non-negative and sum to 1.

        Returns
        -------
        list of cvxpy.constraints
            The constraints for the optimization problem.
        """
        pass

    def __repr__(self):
        return self.__class__.__name__


class WassersteinMixin:
    """
    Mixin class for Wasserstein distance between distributions of causes.

    The Wasserstein distance between two distributions p0 and p1 can be
    computed as a linear program:
        W(p0, p1) = min_{gamma} <gamma, M>
        s.t. (gamma 1 = p0), (gamma^T 1 = p1), (gamma >= 0),
    where M is the matrix of pairwise distances between the atoms of
    the two distributions.
    """

    def __init__(
        self, distributions_of_causes: dict[str, Distribution], lipschitz_l=None
    ):
        self._lipschitz_l = lipschitz_l
        self._distributions_of_causes = distributions_of_causes
        Distribution.homogenize_atoms(self._distributions_of_causes.values())

        # Compute pairwise distances between atoms of the two distributions.
        atoms = next(iter(self._distributions_of_causes.values())).atoms
        # `atoms` is a 2D array of shape (n_atoms, n_causes)
        self._atoms_distances = np.linalg.norm(
            atoms[:, None, :] - atoms[None, :, :], axis=-1, ord=1
        )
        self._gamma = cp.Variable(self._atoms_distances.shape)
        self._L = cp.Parameter((), "l", nonneg=True)

    def _get_wasserstein_constraints(self, target_name, donor_names, weights):
        d_target = self._distributions_of_causes[target_name]
        d_donors = [self._distributions_of_causes[d] for d in donor_names]
        donors_probabilities = np.array([d.probabilities for d in d_donors])
        target_probabilities = np.array(d_target.probabilities)

        p0_hat = donors_probabilities.T @ weights
        p0 = target_probabilities

        constraint_gamma1 = cp.sum(self._gamma, axis=1) == p0_hat
        constraint_gamma2 = cp.sum(self._gamma, axis=0) == p0
        constraint_gamma3 = self._gamma >= 0
        return [constraint_gamma1, constraint_gamma2, constraint_gamma3]

    def _get_wasserstein_l1(self):
        return cp.sum(cp.multiply(self._gamma, self._atoms_distances))

    def get_bound(self):
        # check if self has attribute _objective_function
        if not hasattr(self, "_objective_function"):
            raise ValueError("The estimator is not fitted.")
        return self._objective_function.value


class StandardEstimator(BaseEstimator):
    """Standard synthetic control estimator.

    Minimize the L2 norm between the synthetic control and the treated unit,
    subject to the constraints that the weights are non-negative and sum to 1.
    """

    def _get_constraints(self):
        """The weights must be non-negative and sum to 1."""
        return self._get_constraints_weights_simplex()

    def _get_objective_function(self):
        """The L2 norm between the synthetic control and the treated unit
        outcomes over the pre-treatment period."""
        return self._get_objective_function_l2_pretreatment()


class MBondEstimator(BaseEstimator, WassersteinMixin):
    """M-bond estimator for synthetic control.

    Minimize the Wasserstein distance between the distributions of causes of
    the donors and the target.
    """

    def __init__(self, dataframe, observed_causes, **kwargs):
        super().__init__(dataframe, **kwargs)
        distributions_of_causes, lipschitz_l = (
            observed_causes.get_optimal_distributions_of_causes_and_lipschitz_constant()
        )

        WassersteinMixin.__init__(self, distributions_of_causes)

    def _get_objective_function(self):
        """The Wasserstein distance between the distributions of causes
        of the donors and the target."""
        return self._get_wasserstein_l1()

    def _get_constraints(self):
        return [
            *self._get_constraints_weights_simplex(),
            *self._get_wasserstein_constraints(
                self.target_name, self.donor_names, self._weights
            ),
        ]


class JamesBondEstimator(BaseEstimator, WassersteinMixin):
    """James-bond estimator for synthetic control."""

    def __init__(
        self,
        dataframe,
        observed_causes: ObservedCausesData,
        lipschitz_constant_multiplier=1.0,
        **kwargs
    ):
        super().__init__(dataframe, **kwargs)
        distributions_of_causes, lipschitz_l = (
            observed_causes.get_optimal_distributions_of_causes_and_lipschitz_constant()
        )
        WassersteinMixin.__init__(
            self,
            distributions_of_causes,
            lipschitz_l * lipschitz_constant_multiplier,
        )

    def _get_objective_function(self):
        """The Wasserstein distance between the distributions of causes of the
        donors and the target."""
        wasserstein_l1 = self._get_wasserstein_l1()
        error = self._synthetic_outcomes - self.target_outcomes
        error = error[: self._treatment_start_time_index]
        max_error = cp.max(cp.abs(error))
        self._L.value = self._lipschitz_l
        return max_error + self._L * wasserstein_l1

    def _get_constraints(self):
        return [
            *self._get_constraints_weights_simplex(),
            *self._get_wasserstein_constraints(
                self.target_name, self.donor_names, self._weights
            ),
        ]
