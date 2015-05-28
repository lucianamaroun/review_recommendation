from numpy.random import normal, multivariate_normal
from math import sqrt
import numpy as np

""" Performs Gibbs Sampling over variable_groups.

    Observation: each Variable object has a value and a list of samples. Once a
    new sample is generated, the value is updated to this new sample and the
    subsequent calculations of mean and variance of other values use this new
    value.

    Args:
      variable_groups: a dict of VariableGroup objects.
      n_samples: the number of samples to obtain.

    Returns:
      None. The samples are inserted into Variable objects.
"""
def gibbs_sample(variable_groups, votes, n_samples):
  for _ in xrange(n_samples):
    for variable_group in variable_groups.itervalues():
      for variable_instance in variable_group.iter_instances():
        mean, var = variable_instance.get_cond_mean_and_var(variable_groups,
            votes)
        variable_instance.add_sample(normal(mean, sqrt(var)))
