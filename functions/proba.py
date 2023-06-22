import warnings

import numpy as np
from scipy.stats import multivariate_normal

from river import covariance
from river.proba import base

__all__ = ["MultivariateGaussian"]


class MultivariateGaussian(base.ContinuousDistribution):
    """Multivariate normal distribution with parameters mu and var.

    Parameters
    ----------
    seed
        Random number generator seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd

    >>> np.random.seed(42)
    >>> X = pd.DataFrame(np.random.random((8, 3)),\
                         columns=["red", "green", "blue"])
    >>> X
            red     green      blue
    0  0.374540  0.950714  0.731994
    1  0.598658  0.156019  0.155995
    2  0.058084  0.866176  0.601115
    3  0.708073  0.020584  0.969910
    4  0.832443  0.212339  0.181825
    5  0.183405  0.304242  0.524756
    6  0.431945  0.291229  0.611853
    7  0.139494  0.292145  0.366362

    >>> p = MultivariateGaussian()
    >>> p.n_samples
    0.0
    >>> for x in X.to_dict(orient="records"):
    ...     p = p.update(x)
    >>> p._var
            blue     green    red     
     blue    0.076    0.020   -0.010  
    green    0.020    0.113   -0.053  
      red   -0.010   -0.053    0.079  

    Weighted samples are currently not implemented. Updates with default w = 1.
    >>> p = p.update(x, w=2)

    There are different ways to get current state
    >>> p
    𝒩(μ=(0.385, 0.376, 0.501),
    σ^2=([0.069 0.019 -0.004]
     [0.019 0.100 -0.044]
     [-0.004 -0.044 0.078]))
    >>> p.n_samples
    9.0

    To retrieve pdf and cdf
    # TODO: find out why there is different precission loss locally and on git
    >>> p(x)  # doctest: +ELLIPSIS
    1.70399123552737...
    >>> p.cdf(x)  # doctest: +ELLIPSIS
    0.01421620021072799...

    >>> from river import utils
    >>> p = utils.Rolling(MultivariateGaussian(), window_size=5)
    >>> for x in X.to_dict(orient="records"):
    ...     p = p.update(x)
    >>> p._var
            blue     green    red     
     blue    0.087   -0.023    0.008  
    green   -0.023    0.014   -0.025  
      red    0.008   -0.025    0.095  

    >>> from datetime import datetime as dt, timedelta as td
    >>> X.index = [dt(2023, 3, 28, 0, 0, 0) + td(seconds=x) for x in range(8)]
    >>> p = utils.TimeRolling(MultivariateGaussian(), period=td(seconds=5))
    >>> for t, x in X.iterrows():
    ...     p = p.update(x.to_dict(), t=t)
    >>> p._var
            blue     green    red     
     blue    0.087   -0.023    0.008  
    green   -0.023    0.014   -0.025  
      red    0.008   -0.025    0.095  

    Weighted samples are currently not implemented. Updates with default w = 1.
    >>> p = p.update(x.to_dict(), t=t + td(seconds=1), **{"w":2})

    >>> from river.proba import Gaussian
    >>> p = MultivariateGaussian()
    >>> p_ = Gaussian()
    >>> for t, x in X.iterrows():
    ...     p = p.update(x.to_dict())
    ...     p_ = p_.update(x['blue'])
    >>> p.sigma[0][0] == p_.sigma
    True

    Initiation of class from state is currently not implemented
    >>> p = MultivariateGaussian()._from_state(
    ...     0, 0, 0, 0)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    NotImplementedError: from_state_ not implemented yet.

    """  # noqa: W291

    def __init__(self, seed=None):
        super().__init__()
        self._var = covariance.EmpiricalCovariance(ddof=1)
        self.feature_names_in_ = None

    @classmethod
    def _from_state(cls, n, m, sig, ddof):
        raise NotImplementedError("_from_state not implemented yet.")

    @property
    def n_samples(self):
        if not self._var.matrix:
            return 0.0
        else:
            return list(self._var.matrix.values())[-1].mean.n

    @property
    def mu(self):
        return list({key1: values.mean.get()
                     for (key1, key2), values in self._var.matrix.items()
                     if key1 == key2}.values())

    @property
    def var(self):
        variables = sorted(list({var
                                for cov in self._var.matrix.keys()
                                for var in cov}))
        # Initialize the covariance matrix array
        cov_array = np.zeros((len(variables), len(variables)))

        # Fill in the covariance matrix array
        for i in range(len(variables)):
            for j in range(i, len(variables)):
                if i == j:
                    # Fill in the diagonal with variances
                    cov_array[i, j] = self._var[(variables[i],
                                                 variables[j])].get()
                else:
                    # Fill in the off-diagonal with covariances
                    cov_array[i, j] = self._var[(variables[i],
                                                 variables[j])].get()
                    cov_array[j, i] = self._var[(variables[i],
                                                 variables[j])].get()
        return cov_array

    @property
    def sigma(self):
        cov_array = self.var
        return [[x ** 0.5 if x > 0 else float('nan') for x in row]
                for row in cov_array]

    def __repr__(self):
        mu_str = ', '.join(f'{m:.3f}' for m in self.mu)
        var_str = '\n '.join('[' + ' '.join(f'{s:.3f}' for s in row) + ']'
                             for row in self.var)
        return f"𝒩(μ=({mu_str}),\nσ^2=({var_str}))"

    def update(self, x, w=1.0):
        if w != 1.0:
            warnings.warn("Weights not implemented yet.", RuntimeWarning)
        self._var.update(x)
        return self

    def revert(self, x, w=1.0):
        if w != 1.0:
            # TODO: find out why not called during TimeRolling usage test
            warnings.warn("Weights not implemented yet.",
                          RuntimeWarning)  # pragma: no cover
        self._var.revert(x)
        return self

    def __call__(self, x):
        x = list(x.values())
        var = self.var
        if var is not None:
            try:
                return multivariate_normal(self.mu, var).pdf(x)
            # TODO: validate occurence of ValueError
            except ValueError:  # pragma: no cover
                return 0.0
            # TODO: validate occurence of OverflowError
            except OverflowError:  # pragma: no cover
                return 0.0
        return 0.0  # pragma: no cover

    def cdf(self, x):
        x = list(x.values())
        var = self.var
        try:
            return multivariate_normal(self.mu, var,
                                       allow_singular=True).cdf(x)
        # TODO: validate occurence of ZeroDivisionError
        except ZeroDivisionError:  # pragma: no cover
            return 0.0
