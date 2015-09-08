""" EM module
    ---------------

    Performs EM method to fit latent variables and parameters of CAP. 

    Usage: this module is not directly executed.
"""


from time import time
from math import ceil

from numpy.random import normal, multivariate_normal
from math import sqrt
from numpy import array, identity, zeros
from numpy.linalg import pinv
from multiprocessing import Pool, Manager, log_to_stderr
from pymongo import MongoClient
import logging

from cap.models import ScalarVariable, ArrayVariable, EntityScalarVariable, \
    InteractionScalarGroup, InteractionScalarVariable
from cap import const
from cap.aux import sigmoid


_THREADS = 6
_PARALLEL = True 


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
  # TODO: Put a function to obtain dict from objects
  d_groups = {} 
  for g_id, group in groups.iteritems():
    d_group = {}
    d_group['_id'] = str(g_id)
    d_group['pair_name'] = group.pair_name
    if isinstance(group, InteractionScalarGroup):
      d_group['entity_type_1'] = group.e_type[0]
      d_group['entity_type_2'] = group.e_type[1]
    else:
      d_group['entity_type'] = group.e_type
    d_group['shape'] = group.shape
    d_groups[g_id] = d_group
  d_vars = {}
  for group in groups.itervalues():
    d_vars[group.name] = {}
    for variable in group.iter_variables():
      d_var = {}
      if isinstance(variable, InteractionScalarVariable):
        d_var['related_votes'] = [_id for _id in votes if \
          votes[_id][variable.e_type[0]] == variable.entity_id[0] and \
          votes[_id][variable.e_type[1]] == variable.entity_id[1]]
      else:
        d_var['related_votes'] = [_id for _id in votes if \
          votes[_id][variable.e_type] == variable.entity_id]
      d_var['num_votes'] = len(d_var['related_votes'])
      if isinstance(variable, EntityScalarVariable):
        d_var['type'] = 'EntityScalar'
      elif isinstance(variable, InteractionScalarVariable):
        d_var['type'] = 'InteractionScalar'
      else:
        d_var['type'] = 'EntityArray'
        d_var['last_matrix'] = None
      d_vars[group.name][variable.entity_id] = d_var
  for i in xrange(const.EM_ITER_FIRST):
    print "EM iteration %d" % i
    print "E-step"
    e_time = time()
    perform_e_step(groups, d_groups, votes, d_vars, const.GIBBS_SAMPLES_FIRST)
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
    perform_e_step(groups, d_groups, votes, d_vars, const.GIBBS_SAMPLES_SECOND)
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
    perform_e_step(groups, d_groups, votes, d_vars, const.GIBBS_SAMPLES_THIRD)
    print "E-step Time:\t%f" % (time() - e_time)
    print "M-step"
    m_time = time()
    perform_m_step(groups, votes)
    print "M-step Time:\t%f" % (time() - m_time)
    print "Total:\t\t%f" % (time() - e_time)
    print "------------------------"


def perform_e_step(groups, d_groups, votes, d_vars, n_samples):
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
  d_params = {}
  for group in groups.itervalues():
    param = {}
    param['_id'] = group.name
    param['weight'] = group.weight_param.value
    param['var'] = group.var_param.value
    param['var_H'] = group.var_H.value
    d_params[group.name] = param
#db.params.update_one({'_id': group.name}, {'$set': param}, upsert=True) 
  reset_variables_samples(groups, d_vars)
  d_samples = gibbs_sample(groups, d_groups, votes, d_params, d_vars, n_samples)
  calculate_empiric_mean_and_variance(groups, d_samples)


def reset_variables_samples(groups, d_vars):
  """ Resets sample of variables between EM iterations.

      Args:
        groups: dictionary of Group objects.

      Returns:
        None. The samples of variables are updated (cleaned).
  """
  for group in groups.itervalues():
    for variable in group.iter_variables():
      variable.reset_samples()
      d_vars[group.name][variable.entity_id]['last_sample'] = variable.value
      if 'last_matrix' in d_vars[group.name][variable.entity_id]:
        d_vars[group.name][variable.entity_id]['last_matrix'] = variable.value \
            .dot(variable.value.T)
      if isinstance(variable, ScalarVariable):
        d_vars[group.name][variable.entity_id]['cond_var'] = 1.0 / (1.0 / group.var_param.value + \
          float(d_vars[group.name][variable.entity_id]['num_votes']) / group.var_H.value)
        if isinstance(variable, EntityScalarVariable):
          d_vars[group.name][variable.entity_id]['var_dot'] = group.weight_param.value.T \
            .dot(variable.features)[0,0] / group.var_param.value 
        else:
          d_vars[group.name][variable.entity_id]['var_dot'] = sigmoid(group.weight_param.value.T \
            .dot(variable.features)[0,0]) / group.var_param.value 
      else:
        d_vars[group.name][variable.entity_id]['inv_var'] = pinv(group.var_param.value * identity(const.K))
        d_vars[group.name][variable.entity_id]['var_dot'] = \
            d_vars[group.name][variable.entity_id]['inv_var'].dot(group.weight_param.value) \
            .dot(variable.features)


def gibbs_sample(groups, d_groups, votes, d_params, d_vars, n_samples):
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
  d_samples = {}
  for group in groups.itervalues():
    d_samples[group.name] = {}
    for variable in group.iter_variables():
      d_samples[group.name][variable.entity_id] = []
  for _ in xrange(n_samples):
    for g_name in sorted(groups.keys()):
      group = groups[g_name]
      variables = [var.entity_id for var in group.iter_variables()]
      res = []
      if _PARALLEL:
        #print "--> Starting job of group %s" % g_name
        chunk_size = int(ceil(group.get_size() / float(_THREADS)))
        #print "# variables: %d" % group.get_size()
        #print "chunk size: %d" % chunk_size 
        n = group.get_size()
        pool = Pool(processes=_THREADS)
        for i in xrange(_THREADS):
          if i*chunk_size >= n:
            continue
         # res.append(sample_variables(g_name, 
         #     variables[i*chunk_size:(i+1)*chunk_size], 
         #     d_groups, d_params, d_vars, votes))
          res.append(pool.apply_async(sample_variables, 
              (g_name, variables[i*chunk_size:(i+1)*chunk_size], d_groups,
              d_params, d_vars, votes)))
        pool.close()
        pool.join()
        for r in res:
          r = r.get()
          for _id in r:
            sample = r[_id]
            d_samples[g_name][_id].append(sample)
            d_vars[g_name][_id]['last_sample'] = sample
            if 'last_matrix' in d_vars[g_name][_id]:
              d_vars[g_name][_id]['last_matrix'] = sample.dot(sample.T)
        #print "--> Ended job of group %s" % g_name
      else:
        res.append(sample_variables(g_name, 
            variables, 
            d_groups, d_params, d_vars, votes))
        for r in res:
          for _id in r:
            sample = r[_id]
            d_samples[g_name][_id].append(sample)
            d_vars[g_name][_id]['last_sample'] = sample
            if 'last_matrix' in d_vars[g_name][_id]:
              d_vars[g_name][_id]['last_matrix'] = sample.dot(sample.T)
        #for variable in group.iter_variables():
         # if isinstance(variable, ScalarVariable):
         #   mean, var = variable.get_cond_mean_and_var(groups, votes)
         #   sample = normal(mean, sqrt(var))
         # else:
         #   mean, cov = variable.get_cond_mean_and_var(groups, votes)
         #   mean = mean.reshape(-1)
         #   sample = multivariate_normal(mean, cov)
         # variable.add_sample(sample)
  return d_samples

def get_rest_value(name, vote, d_groups, d_vars):
  """ Gets the value of the variable as the truth minus all the other terms
      except the one involving this variable, for a given vote.

      Args:
        groups: dictionary of groups.
        vote: dictionary of a modeled vote.

      Returns:
        The value of rest, which is equal to the truth value minus all other
      terms.
  """    
  rest = vote['vote']
  self_group = d_groups[name] 
  for group in d_groups.values(): #TODO: itervalues
    if group['_id'] == name or group['_id'] == self_group['pair_name']:
      continue
    if 'entity_type_1' in group:
      key = (vote[group['entity_type_1']], vote[group['entity_type_2']])
    else: 
      key = vote[group['entity_type']]
    if key not in d_vars[group['_id']]:
      continue
    var_value = d_vars[group['_id']][key]['last_sample']
    pair_name = group['pair_name']
    if pair_name:
      if group['_id'] > pair_name:
        continue
      pair_group = d_groups[pair_name] 
      pair_value = d_vars[pair_name][vote[pair_group['entity_type']]] \
          ['last_sample']
      rest = rest - var_value.T.dot(pair_value)[0,0]
    else:
      rest = rest - var_value
  return rest


def sample_variables(name, var_ids, d_groups, d_params, d_vars, votes):
  samples = {}
  param = d_params[name]
  for var_id in var_ids:
    d_var = d_vars[name][var_id]
    if d_var['type'] != 'EntityArray':
      variance = 0.0
      mean = 0.0
      for i in d_var['related_votes']:
        vote = votes[i] 
        rest = get_rest_value(name, vote, d_groups, d_vars)
        mean += rest
      mean /= param['var_H']
      var = d_var['cond_var']  
      mean = var * (mean + d_var['var_dot'])
      sample = normal(mean, sqrt(var))
    else:
      variance = zeros((const.K, const.K))
      mean = zeros((const.K, 1))
      group = d_groups[name] 
      pair_group = d_groups[group['pair_name']] 
      for i in d_var['related_votes']:
        vote = votes[i] 
        rest = get_rest_value(name, vote, d_groups, d_vars)
        pair_var = d_vars[group['pair_name']][vote[pair_group['entity_type']]]
        variance += pair_var['last_matrix'] 
        mean += rest * pair_var['last_sample']
      mean /= param['var_H']
      cov = pinv(variance / param['var_H'] + d_var['inv_var'])
      mean = cov.dot(d_var['var_dot'] + mean)
      mean = mean.reshape(-1)
      sample = multivariate_normal(mean, cov).reshape(group['shape'])
    samples[var_id] = sample
  return samples 

def calculate_empiric_mean_and_variance(groups, d_samples):
  """ Calculates empiric mean and variance of the groups from samples.

      Args:
        groups: dictionary of Group of variables objects.

      Returns:
        None. The values of mean and variance are updated on each Variable
      object.
  """
  for group in groups.itervalues():
    for variable in group.iter_variables():
      #if _PARALLEL:
      variable.samples = d_samples[variable.name][variable.entity_id]
      variable.num_samples = len(variable.samples)
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
