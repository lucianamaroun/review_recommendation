from numpy.random import normal, multivariate_normal
from math import sqrt
import numpy as np

from cap.models import ScalarVariable, ArrayVariable

""" Performs Gibbs Sampling over groups.

    Observation: each Variable object has a value and a list of samples. Once a
    new sample is generated, the value is updated to this new sample and the
    subsequent calculations of mean and variance of other values use this new
    value.

    Args:
      groups: a dict of Group objects.
      n_samples: the number of samples to obtain.

    Returns:
      None. The samples are inserted into Variable objects.
"""
def gibbs_sample(groups, votes, n_samples):
  for _ in xrange(n_samples):
    for group in groups.itervalues():
      for variable in group.iter_variables():
        if isinstance(variable, ScalarVariable):
          mean, var = variable.get_cond_mean_and_var(groups, votes)
          variable.add_sample(normal(mean, sqrt(var)))
        if isinstance(variable, ArrayVariable):
          mean, cov = variable.get_cond_mean_and_var(groups, votes)
          variable.add_sample(multivariate_normal(mean, cov))
