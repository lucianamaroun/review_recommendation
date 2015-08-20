""" EM module
    ---------------

    Performs EM method to fit latent variables and parameters of CAP. 

    Usage: this module is not directly executed.
"""

from time import time

from numpy.random import normal, multivariate_normal
from math import sqrt
import numpy as np

from cap.models import ScalarVariable, ArrayVariable
from cap import const

def expectation_maximization(groups, votes):
  """ Expectation Maximization algorithm for CAP baseline. Iterates over E and
      M-steps, fitting latent variables and parameters, respectively.
      
      Observation: The number of iterations was split in three stages. Each one
        has a number of gibbs samples to compute. This was suggested in RLFM
        paper (http://dl.acm.org/citation.cfm?id=1557029).

      Args:
        groups: dictionary of Group of variables objects.
        votes: list of votes, represented as dictionaries, which is the data.
       
      Returns:
        None. Variable and Parameter objects are changes in place.
  """ 
  for i in xrange(const.EM_ITER_FIRST):
    print "EM iteration %d" % i
    print "E-step ",
    e_time = time()
    perform_e_step(groups, votes, const.GIBBS_SAMPLES_FIRST)
    print time() - e_time
    print "M-step ",
    m_time = time()
    perform_m_step(groups, votes)
    print time() - m_time
    print "Total %f" % time() - e_time
    print "------------------------"
  for i in xrange(const.EM_ITER_SECOND):
    print "EM iteration %d" % (const.EM_ITER_FIRST + i)
    print "E-step ",
    e_time = time()
    perform_e_step(groups, votes, const.GIBBS_SAMPLES_SECOND)
    print time() - e_time
    print "M-step ",
    m_time = time()
    perform_m_step(groups, votes)
    print time() - m_time
    print "Total %f" % time() - e_time
    print "------------------------"
  for i in xrange(const.EM_ITER_THIRD):
    print "EM iteration %d" % (const.EM_ITER_FIRST + const.EM_ITER_SECOND + i)
    print "E-step ",
    e_time = time()
    perform_e_step(groups, votes, const.GIBBS_SAMPLES_THIRD)
    print time() - e_time
    print "M-step ",
    m_time = time()
    perform_m_step(groups, votes)
    print time() - m_time
    print "Total %f" % time() - e_time
    print "------------------------"


def perform_e_step(groups, votes, n_samples):
  """ Performs E-step of EM algorithm. Consists of calculating the expectation
      of the complete log-likelihood with respect to the posterior of latent
      variables (distribution of latent variables given data and parameters).

      Observation: The E value is not in closed format, thus is approximated
      by gibbs sampling.
  
      Args:
        groups: dictionary of Group of variables objects.
        votes: list of votes, represented as dictionaries, which is the data.
        n_samples: number of gibbs samples to perform to approximate the
          expectation.
 
      Returns:
        None. The variables will have samples, empiric mean and variance
          attributes updated. 
  """
  reset_variables_samples(groups)
  print "Gibbs Sampling"
  gibbs_sample(groups, votes, n_samples)
  print "Calculation of Empiric Stats"
  calculate_empiric_mean_and_variance(groups)


def reset_variables_samples(groups):
  """ Resets sample of variables between EM iterations.

      Args:
        groups: dictionary of Group objects.

      Returns:
        None. The samples of variables are updated (cleaned).
  """
  for group in groups.itervalues():
    for variable in group.iter_variables():
      variable.reset_samples()


def gibbs_sample(groups, votes, n_samples):
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
  for _ in xrange(n_samples):
    for g_name in sorted(groups.keys()):
      group = groups[g_name]
      for variable in group.iter_variables():
          # make it a polymorphic function and avoid isinstance
        if isinstance(variable, ScalarVariable):
          mean, var = variable.get_cond_mean_and_var(groups, votes)
          variable.add_sample(normal(mean, sqrt(var)))
        if isinstance(variable, ArrayVariable):
          mean, cov = variable.get_cond_mean_and_var(groups, votes)
          mean = mean.reshape(-1)
          sample = multivariate_normal(mean, cov).reshape(mean.size, 1)
          variable.add_sample(sample)


def calculate_empiric_mean_and_variance(groups):
  """ Calculates empiric mean and variance of the groups from samples.

      Args:
        groups: dictionary of Group of variables objects.

      Returns:
        None. The values of mean and variance are updated on each Variable
      object.
  """
  for group in groups.itervalues():
    for variable in group.iter_variables():
      variable.calculate_empiric_mean()
      variable.calculate_empiric_var()


def perform_m_step(groups, votes):
  """ Performs M-step of EM algorithm. Adjust parameters by OLS, fitting them
      as linear regression weights for monte carlo means calculated on E-step.

      Args:
        groups: dictionary of Group objects, indexed by name.
        votes: list of vote dictionaries, the data.

      Returns:
        None. The values of the Parameter objects are updated.
  """
  for group in groups.itervalues():
    group.weight_param.optimize(group)
    group.var_param.optimize(group)
  groups.itervalues().next().var_H.optimize(groups, votes)
