""" EM module
    ---------

    Performs EM method to fit latent variables and parameters of CAP. 

    Usage: this module is not directly callable.
"""


from time import time
from math import ceil
from time import time
from random import shuffle

from numpy.random import normal, multivariate_normal, seed
from math import sqrt
from numpy import array, identity, zeros, ones, isnan
from numpy.linalg import pinv
from multiprocessing import Pool

from algo.cap.models import ScalarVariable, ArrayVariable, EntityScalarVariable, \
    InteractionScalarGroup, InteractionScalarVariable, EntityArrayGroup
from algo.cap import const
from util.aux import sigmoid


def expectation_maximization(groups, votes):
  """ Expectation Maximization algorithm for CAP baseline. Iterates over E and
      M-steps, fitting latent variables and parameters, respectively.
      
      Observation:
      - The number of iterations was split in three stages. Each one has a 
      number of gibbs samples to compute. This was suggested in RLFM paper
      (http://dl.acm.org/citation.cfm?id=1557029).

      Args:
        groups: dictionary of Group of variables objects.
        votes: list of votes, each one represented as a dictionary, which is the
      training data.
       
      Returns:
        None. Variable and Parameter objects are changes in place.
  """
  for stage, num_iter in enumerate(const.EM_ITER):
    print 'Stage %d' % stage
    for i in xrange(num_iter):
      print 'EM iteration %d' % i
      print 'E-step'
      e_time = time()
      perform_e_step(groups, votes, const.SAMPLES[stage], const.BURN_IN[stage])
      print 'E-step Time:\t%f' % (time() - e_time)
      print 'M-step'
      m_time = time()
      perform_m_step(groups, votes)
      print 'M-step Time:\t%f' % (time() - m_time)
      print 'Total:\t\t%f' % (time() - e_time)
      print '------------------------'


def perform_e_step(groups, votes, n_samples, n_burnin):
  """ Performs E-step of EM algorithm. Consists of calculating the expectation
      of the complete log-likelihood with respect to the posterior of latent
      variables (distribution of latent variables given data and parameters).

      Observations: 
      - The E value is not in closed format, thus is approximated by Gibbs 
      Sampling.
       
      Args:
        groups: dictionary of Group of variables objects.
        votes: list of votes, each one represented as a dictionary, which is the
      training data.
        n_samples: number of gibbs samples to perform to approximate the
          expectation.
        n_burnin: number of burn in samples, that is, amount of initial samples
          to ignore.
 
      Returns:
        None. The variables will have samples, empiric mean and variance
      attributes updated. 
  """
  reset_variables_samples(groups)
  gibbs_sample(groups, votes, n_samples, n_burnin)
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


def gibbs_sample(groups, votes, n_samples, n_burnin):
  """ Performs Gibbs Sampling over groups.

      Observations:
      - Each Variable object has a list of samples. Once a new sample is
      generated, the value used for the variable is the new sample.
      - After sampling, the values of latent variables are changed to empiric
      mean of samples.

      Args:
        groups: a dict of Group objects.
        votes: list of votes, each one represented as a dictionary, which is the
      training data.
        n_samples: the number of samples to obtain.
        n_burnin: number of initial samples to ignore.

      Returns:
        None. The samples are inserted into Variable objects.
  """
  burn_count = 0
  for _ in xrange(n_samples + n_burnin):
    for g_name in ['alpha', 'beta', 'xi', 'u', 'v', 'gamma', 'lambda']:
      group = groups[g_name]
      if isinstance(group, EntityArrayGroup):
        for variable in group.iter_variables():
          mean, var = variable.get_cond_mean_and_var(groups, votes)
          sample = multivariate_normal(mean.reshape(-1), var)
          variable.add_sample(sample.reshape(sample.size, 1))
      else:
        for variable in group.iter_variables():
          mean, var = variable.get_cond_mean_and_var(groups, votes)
          sample = normal(mean, sqrt(var))
          variable.add_sample(sample)
      if burn_count < n_burnin:
        burn_count += 1


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
