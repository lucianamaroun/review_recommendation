""" EM module
    ---------------

    Performs EM method to fit latent variables and parameters of CAP. 

    Usage: this module is not directly executed.
"""


from time import time
from math import ceil

from numpy.random import normal, multivariate_normal
from math import sqrt
import numpy as np
from multiprocessing import Pool, Manager
from pymongo import MongoClient

from cap.models import ScalarVariable, ArrayVariable, EntityScalarVariable, \
    InteractionScalarGroup, InteractionScalarVariable

from cap import const


_THREADS = 4

client = MongoClient()
db = client.sample
db.votes.drop()
db.groups.drop()
db.variables.drop()
db.params.drop()


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
  # Putting votes in MongoDB
  for v_id, vote in votes.iteritems():
    vote['_id'] = str(v_id)
    db.votes.insert_one(vote)
  # Putting groups in MongoDB
  print groups
  for g_id, group in groups.iteritems():
    d_group = {}
    d_group['_id'] = str(g_id)
    d_group['pair_name'] = group.pair_name
    d_group['size'] = group.size
    if isinstance(group, InteractionScalarGroup):
      d_group['entity_type_1'] = group.e_type[0]
      d_group['entity_type_2'] = group.e_type[1]
    else:
      d_group['entity_type'] = group.e_type
    d_group['shape'] = group.shape
    db.groups.insert_one(d_group)
  # Putting variables in MongoDB
  for group in groups.itervalues():
    for variable in group.iter_variables():
      d_var = {}
      if isinstance(variable, InteractionScalarVariable):
        d_var['entity_id_1'] = variable.entity_id[0]
        d_var['entity_id_2'] = variable.entity_id[1]
      else:
        d_var['entity_id'] = variable.entity_id
      d_var['group'] = variable.name
      d_var['last_sample'] = variable.value if isinstance(variable,
          ScalarVariable) else variable.value.reshape(-1).tolist()
      d_var['samples'] = []
      db.variables.insert_one(d_var)
  for i in xrange(const.EM_ITER_FIRST):
    print "EM iteration %d" % i
    print "E-step"
    e_time = time()
    perform_e_step(groups,  votes, const.GIBBS_SAMPLES_FIRST)
    print "E-step Time:\t%f" % (time() - e_time)
    print "M-step"
    m_time = time()
    perform_m_step(groups, votes)
    print "M-step Time:\t%f" % (time() - m_time)
    print "Total:\t\t%f" % (time() - e_time)
    print "------------------------"
  for i in xrange(const.EM_ITER_SECOND):
    print "EM iteration %d" % (const.EM_ITER_FIRST + i)
    print "E-step"
    e_time = time()
    perform_e_step(groups, votes, const.GIBBS_SAMPLES_SECOND)
    print "E-step Time:\t%f" % (time() - e_time)
    print "M-step"
    m_time = time()
    perform_m_step(groups, votes)
    print "M-step Time:\t%f" % (time() - m_time)
    print "Total:\t\t%f" % (time() - e_time)
    print "------------------------"
  for i in xrange(const.EM_ITER_THIRD):
    print "EM iteration %d" % (const.EM_ITER_FIRST + const.EM_ITER_SECOND + i)
    print "E-step"
    e_time = time()
    perform_e_step(groups, votes, const.GIBBS_SAMPLES_THIRD)
    print "E-step Time:\t%f" % (time() - e_time)
    print "M-step"
    m_time = time()
    perform_m_step(groups, votes)
    print "M-step Time:\t%f" % (time() - m_time)
    print "Total:\t\t%f" % (time() - e_time)
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
  # Putting params in mongoDB
  for group in groups.itervalues():
    param = {}
    param['_id'] = group.name
    param['weight'] = group.weight_param.value.reshape(-1).tolist()
    param['var'] = group.var_param.value
    param['var_H'] = group.var_H.value
    db.params.update_one({'_id': group.name}, {'$set': param}, upsert=True) 
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
      print "--> Starting job of group %s" % g_name
      group = groups[g_name]
      chunk_size = int(ceil(group.get_size() / float(_THREADS)))
      print "# variables: %d" % group.get_size()
      print "chunk size: %d" % chunk_size 
      n = group.get_size()
     # groups = manager.dict(groups)
     # votes = manager.dict(votes)
     # pool = Pool(processes=_THREADS) 
     # pool.map(sample_variable, group.iter_variables(), chunk_size)
     # pool.close()
     # pool.join()
      for variable in group.iter_variables():
        sample_variable(variable)
      print "--> Ended job of group %s" % g_name

def sample_variable(variable):
  if isinstance(variable, EntityScalarVariable):
    mean, var = variable.get_cond_mean_and_var()
    sample = normal(mean, sqrt(var))
    #variable.add_sample(sample)
    d_var = db.variables.find_one({'group': variable.name, 'entity_id':
        variable.entity_id})
    print variable.entity_id
    print d_var
    d_var['samples'].append(sample)
    d_var['last_sample'] = sample
    db.variables.update_one({'group': variable.name, 'entity_id': variable.entity_id}, 
        {'$set': {'sample': d_var['samples'], 'last_sample':
        d_var['last_sample']}})
  elif isinstance(variable, InteractionScalarVariable):
    mean, var = variable.get_cond_mean_and_var()
    sample = normal(mean, sqrt(var))
    #variable.add_sample(sample)
    d_var = db.variables.find_one({'group': variable.name, 'entity_id_1':
        variable.entity_id[0], 'entity_id_2': variable.entity_id[1]})
    d_var['samples'].append(sample)
    d_var['last_sample'] = sample
    db.variables.update_one({'group': variable.name, 'entity_id_1':
        variable.entity_id[0], 'entity_id_2': variable.entity_id[1]}, 
        {'$set': {'sample': d_var['samples'], 'last_sample':
        d_var['last_sample']}})
  elif isinstance(variable, ArrayVariable):
    mean, cov = variable.get_cond_mean_and_var()
    mean = mean.reshape(-1)
    sample = multivariate_normal(mean, cov)
    #variable.add_sample(sample)
    d_var = db.variables.find_one({'group': variable.name, 'entity_id':
        variable.entity_id})
    d_var['samples'].append(sample)
    d_var['last_sample'] = sample
    db.variables.update_one({'group': variable.name, 'entity_id': variable.entity_id}, 
        {'$set': {'sample': d_var['samples'], 'last_sample':
        d_var['last_sample']}})

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
      d_var = db.variables.find_one({'group': variable.name, 'entity_id':
        variable.entity_id})
      variable.samples = d_var.samples
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
