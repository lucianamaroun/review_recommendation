""" Test of EM
    ----------

    Test updates and convergence of EM process. 

    Usage:
    $ python -m cap.test_em
"""


from unittest import TestCase, main
from numpy import array, identity, absolute
from numpy.linalg import det, pinv
from math import log

from algorithms.cap import models, const, em
from util import aux


def likelihood(groups, votes):
  """ Gets the log-likelihood of the current set of variables and parameters.

      Observation: Auxiliary function for testing.

      Args:
        groups: dictionary of Group objects indexed by names.
        votes: dictionary of votes of training set.

      Returns:
        A flot with the log-likelihood value.
  """
  likelihood = 0
  var_H = groups.itervalues().next().var_H.value
  for vote in votes:
    term = vote['vote'] - \
        groups['alpha'].get_instance(vote).value - \
        groups['beta'].get_instance(vote).value - \
        groups['xi'].get_instance(vote).value - \
        groups['u'].get_instance(vote).value.T \
        .dot(groups['v'].get_instance(vote).value)
    if groups['gamma'].contains(vote):
      term += groups['gamma'].get_instance(vote).value
    if groups['lambda'].contains(vote):
      term += groups['lambda'].get_instance(vote).value
    likelihood += term ** 2 / var_H + log(var_H)
  for group in groups.itervalues():
    if isinstance(group, models.EntityScalarGroup):
      for variable in group.iter_variables():
        likelihood += ((variable.value - \
            group.weight_param.value.T.dot(variable.features)) ** 2) / \
            group.var_param.value
        likelihood += log(group.var_param.value)
    elif isinstance(group, models.EntityArrayGroup):
      for variable in group.iter_variables():
        term = variable.value - group.weight_param.value.dot(variable.features)
        covar = group.var_param.value * identity(variable.shape[0])
        likelihood += term.T.dot(covar).dot(term)
        likelihood += log(det(covar))
    else:
      for variable in group.iter_variables():
        likelihood += ((variable.value - \
            aux.sigmoid(group.weight_param.value.T.dot(variable.features))) \
            ** 2) / group.var_param.value
        likelihood += log(group.var_param.value)
  likelihood *= - 1/2
  return likelihood
    

class TinyScenarioTestCase(TestCase):
  """ Test case for a tiny scenario of EM. """

  def setUp(self):
    self.reviews = {
        'r1': array([50, 3, 0.5, 16.67, 1.0, 0.2, 0.3, 0.2, 0.6, 0.3, 0.01,
            0.003, 0.0, 0.02, 0.123, 0.4, 0.3]),
        'r2': array([100, 12, 0.62, 8.33, 0.9, 0.1, 0.4, 0.5, 0.4, 0.2, 0.0,
            0.0, 0.0, 0.2, 0.01, 0.2, 0.7]),
    }
    self.authors = {
      'a1': array([23, 3.7, 10, 4, 0.5]),
    }
    self.voters = {
      'v1': array([12, 17, 0.02, 3.8, 3.5, 3.9, 4.3, 4.5, 4.0]),
    }
    self.sim = {
      ('a1', 'v1'): array([3, 0.2, 0.4, 0.3, 1.6, 2, 3]),
    }
    self.conn = {
      ('a1', 'v1'): array([0.02, 0.01, 0.2, 0.03]),
    }
    self.votes = [
        {'review': 'r1', 'author': 'a1', 'voter': 'v1', 'vote': 4},
        {'review': 'r2', 'author': 'a1', 'voter': 'v1', 'vote': 5},
    ]
    self.groups = {}
    self.var_H = models.PredictionVarianceParameter('var_H')
    self.d_groups = {} 
    self.d_vars = {}
    self.d_params = {}
    self.votes_dict = {i:v for i,v in enumerate(self.votes)}
    self._create_groups()
  
  def _create_groups(self):
    self.groups['alpha'] = models.EntityScalarGroup('alpha', 'voter',
        models.EntityScalarParameter('d', (9,1)), 
        models.ScalarVarianceParameter('var_alpha'),
        self.var_H)
    for e_id, e_feat in self.voters.iteritems():
      self.groups['alpha'].add_instance(e_id, e_feat)
    self.groups['beta'] = models.EntityScalarGroup('beta', 'review',
        models.EntityScalarParameter('g', (17,1)),
        models.ScalarVarianceParameter('var_beta'),
        self.var_H)
    for e_id, e_feat in self.reviews.iteritems():
      self.groups['beta'].add_instance(e_id, e_feat)
    self.groups['xi'] = models.EntityScalarGroup('xi', 'author',
        models.EntityScalarParameter('b', (5,1)),
        models.ScalarVarianceParameter('var_xi'),
        self.var_H)
    for e_id, e_feat in self.authors.iteritems():
      self.groups['xi'].add_instance(e_id, e_feat)
    self.groups['u'] = models.EntityArrayGroup('u', (const.K, 1), 'voter',
        models.EntityArrayParameter('W', (const.K, 9)),
        models.ArrayVarianceParameter('var_u'),
        self.var_H)
    for e_id, e_feat in self.voters.iteritems():
      self.groups['u'].add_instance(e_id, e_feat)
    self.groups['v'] = models.EntityArrayGroup('v', (const.K, 1), 'review',
        models.EntityArrayParameter('V', (const.K,17)),
        models.ArrayVarianceParameter('var_v'),
        self.var_H)
    for e_id, e_feat in self.reviews.iteritems():
      self.groups['v'].add_instance(e_id, e_feat)
    self.groups['gamma'] = models.InteractionScalarGroup('gamma', ('author', 
        'voter'), models.InteractionScalarParameter('r', (7, 1)), 
        models.ScalarVarianceParameter('var_gamma'), self.var_H)
    for e_id, e_feat in self.sim.iteritems():
      self.groups['gamma'].add_instance(e_id, e_feat)
    self.groups['lambda'] = models.InteractionScalarGroup('lambda', ('author',
        'voter'), models.InteractionScalarParameter('h', (4,1)),
        models.ScalarVarianceParameter('var_lambda'), self.var_H)
    for e_id, e_feat in self.conn.iteritems():
      self.groups['lambda'].add_instance(e_id, e_feat)
    self.groups['u'].set_pair_name('v')
    self.groups['v'].set_pair_name('u')
    for g_id, group in self.groups.iteritems():
      d_group = {}
      d_group['_id'] = str(g_id)
      d_group['pair_name'] = group.pair_name
      d_group['size'] = group.size
      d_group['entity_type'] = group.e_type
      d_group['shape'] = group.shape
      self.d_groups[g_id] = d_group
    for group in self.groups.itervalues():
      self.d_vars[group.name] = {}
      for variable in group.iter_variables():
        d_var = {}
        if isinstance(variable, models.InteractionScalarVariable):
          d_var['related_votes'] = [_id for _id in self.votes_dict if \
            self.votes_dict[_id][variable.e_type[0]] == variable.entity_id[0] and \
            self.votes_dict[_id][variable.e_type[1]] == variable.entity_id[1]]
        else:
          d_var['related_votes'] = [_id for _id in self.votes_dict if \
            self.votes_dict[_id][variable.e_type] == variable.entity_id]
        d_var['entity_id'] = variable.entity_id
        d_var['num_votes'] = len(d_var['related_votes'])
        if isinstance(variable, models.ScalarVariable):
          d_var['cond_var'] = 1.0 / (1.0 / group.var_param.value + \
            float(d_var['num_votes']) / group.var_H.value)
        if isinstance(variable, models.EntityScalarVariable):
          d_var['type'] = 'EntityScalar'
          d_var['var_dot'] = group.weight_param.value.T \
            .dot(variable.features)[0,0] / group.var_param.value 
        elif isinstance(variable, models.InteractionScalarVariable):
          d_var['type'] = 'InteractionScalar'
          d_var['var_dot'] = aux.sigmoid(group.weight_param.value.T \
            .dot(variable.features)[0,0]) / group.var_param.value 
        else:
          d_var['type'] = 'EntityArray'
          d_var['inv_var'] = pinv(group.var_param.value * identity(const.K))
          d_var['var_dot'] = d_var['inv_var'].dot(group.weight_param.value) \
              .dot(variable.features)
          d_var['last_matrix'] = variable.value.dot(variable.value.T)
        d_var['group'] = variable.name
        d_var['last_sample'] = variable.value
        d_var['samples'] = []
        self.d_vars[group.name][variable.entity_id] = d_var
    for group in self.groups.itervalues():
      param = {}
      param['_id'] = group.name
      param['weight'] = group.weight_param.value
      param['var'] = group.var_param.value
      param['var_H'] = group.var_H.value
      self.d_params[group.name] = param

  def test_rest_alpha(self):
    var = self.groups['alpha'].iter_variables().next()
    for vote in self.votes:
      rest = vote['vote'] - self.groups['beta'].get_instance(vote).value - \
          self.groups['xi'].iter_variables().next().value - \
          self.groups['gamma'].iter_variables().next().value - \
          self.groups['lambda'].iter_variables().next().value - \
          self.groups['u'].iter_variables().next().value.T \
          .dot(self.groups['v'].get_instance(vote).value)[0,0]
      self.assertAlmostEqual(var.get_rest_value(self.groups, vote), rest) 
      self.assertTrue(rest <= 5)
  
  def test_rest_beta(self):
    iterator = self.groups['beta'].iter_variables()
    var = iterator.next()
    vote = self.votes[0]
    rest = vote['vote'] - self.groups['alpha'].iter_variables().next().value - \
        self.groups['xi'].iter_variables().next().value - \
        self.groups['gamma'].iter_variables().next().value - \
        self.groups['lambda'].iter_variables().next().value - \
        self.groups['u'].iter_variables().next().value.T \
        .dot(self.groups['v'].get_instance(vote).value)[0,0]
    self.assertAlmostEqual(var.get_rest_value(self.groups, vote), rest)
    self.assertTrue(rest <= 5)
    var = iterator.next()
    vote = self.votes[1]  
    rest = vote['vote'] - self.groups['alpha'].iter_variables().next().value - \
        self.groups['xi'].iter_variables().next().value - \
        self.groups['gamma'].iter_variables().next().value - \
        self.groups['lambda'].iter_variables().next().value - \
        self.groups['u'].iter_variables().next().value.T \
        .dot(self.groups['v'].get_instance(vote).value)[0,0]
    self.assertAlmostEqual(var.get_rest_value(self.groups, vote), rest)
    self.assertTrue(rest <= 5)
  
  def test_rest_xi(self):
    var = self.groups['xi'].iter_variables().next()
    for vote in self.votes:
      rest = vote['vote'] - self.groups['beta'].get_instance(vote).value - \
          self.groups['alpha'].iter_variables().next().value - \
          self.groups['gamma'].iter_variables().next().value - \
          self.groups['lambda'].iter_variables().next().value - \
          self.groups['u'].iter_variables().next().value.T \
          .dot(self.groups['v'].get_instance(vote).value)[0,0]
      self.assertAlmostEqual(var.get_rest_value(self.groups, vote), rest) 
      self.assertTrue(rest <= 5)
      
  def test_rest_u(self):
    var = self.groups['u'].iter_variables().next()
    for vote in self.votes:
      rest = vote['vote'] - self.groups['beta'].get_instance(vote).value - \
          self.groups['alpha'].iter_variables().next().value - \
          self.groups['xi'].iter_variables().next().value - \
          self.groups['gamma'].iter_variables().next().value - \
          self.groups['lambda'].iter_variables().next().value
      self.assertAlmostEqual(var.get_rest_value(self.groups, vote), rest) 
      self.assertTrue(rest <= 5)
      
  def test_rest_v(self):
    iterator = self.groups['v'].iter_variables()
    var = iterator.next()
    vote = self.votes[0]
    rest = vote['vote'] - self.groups['alpha'].iter_variables().next().value - \
        self.groups['beta'].get_instance(vote).value - \
        self.groups['xi'].iter_variables().next().value - \
        self.groups['gamma'].iter_variables().next().value - \
        self.groups['lambda'].iter_variables().next().value 
    self.assertAlmostEqual(var.get_rest_value(self.groups, vote), rest)
    self.assertTrue(rest <= 5)
    var = iterator.next()
    vote = self.votes[1]  
    rest = vote['vote'] - self.groups['alpha'].iter_variables().next().value - \
        self.groups['beta'].get_instance(vote).value - \
        self.groups['xi'].iter_variables().next().value - \
        self.groups['gamma'].iter_variables().next().value - \
        self.groups['lambda'].iter_variables().next().value 
    self.assertAlmostEqual(var.get_rest_value(self.groups, vote), rest)
    self.assertTrue(rest <= 5)
  
  def test_rest_gamma(self):
    var = self.groups['gamma'].iter_variables().next()
    for vote in self.votes:
      rest = vote['vote'] - self.groups['beta'].get_instance(vote).value - \
          self.groups['alpha'].iter_variables().next().value - \
          self.groups['xi'].iter_variables().next().value - \
          self.groups['lambda'].iter_variables().next().value - \
          self.groups['u'].iter_variables().next().value.T \
          .dot(self.groups['v'].get_instance(vote).value)[0,0]
      self.assertAlmostEqual(var.get_rest_value(self.groups, vote), rest) 
      self.assertTrue(rest <= 5)
  
  def test_rest_xi(self):
    var = self.groups['lambda'].iter_variables().next()
    for vote in self.votes:
      rest = vote['vote'] - self.groups['beta'].get_instance(vote).value - \
          self.groups['alpha'].iter_variables().next().value - \
          self.groups['gamma'].iter_variables().next().value - \
          self.groups['xi'].iter_variables().next().value - \
          self.groups['u'].iter_variables().next().value.T \
          .dot(self.groups['v'].get_instance(vote).value)[0,0]
      self.assertAlmostEqual(var.get_rest_value(self.groups, vote), rest) 
      self.assertTrue(rest <= 5)
    
  def test_cond_mean_var_alpha(self):
    prev_lkl = likelihood(self.groups, self.votes)
    var = self.groups['alpha'].iter_variables().next()
    var.cond_var = None
    var.var_dot = None
    mean_res, var_res = var.get_cond_mean_and_var(self.groups, self.votes_dict)
    variance = 1.0 / (2.0 / self.groups['alpha'].var_H.value +
        1.0 / self.groups['alpha'].var_param.value)
    mean = variance * ((var.get_rest_value(self.groups, self.votes[0]) + \
        var.get_rest_value(self.groups, self.votes[1])) / \
        self.groups['alpha'].var_H.value + \
        self.groups['alpha'].weight_param.value.T.dot(var.features)[0,0] / \
        self.groups['alpha'].var_param.value)
    self.assertAlmostEqual(var_res, variance)
    self.assertAlmostEqual(mean_res, mean)
    old_value = var.value
    var.update(float(mean_res))
    self.assertGreaterEqual(likelihood(self.groups, self.votes), prev_lkl)
         
  def test_cond_mean_var_beta(self):
    iterator = self.groups['beta'].iter_variables()
    sse = 0
    prev_sse = 0
    for vote in self.votes:
      var = iterator.next()
      var.cond_var = None
      var.var_dot = None
      mean_res, var_res = var.get_cond_mean_and_var(self.groups, self.votes_dict)
      variance = 1.0 / (1.0 / self.groups['beta'].var_H.value +
          1.0 / self.groups['beta'].var_param.value)
      mean = variance * (var.get_rest_value(self.groups, vote) / \
          self.groups['beta'].var_H.value + \
          self.groups['beta'].weight_param.value.T.dot(var.features)[0,0] / \
          self.groups['beta'].var_param.value)
      self.assertAlmostEqual(mean_res, mean)
      self.assertAlmostEqual(var_res, variance)
      prev_lkl = likelihood(self.groups, self.votes)
      var.update(float(mean_res))
      self.assertGreaterEqual(likelihood(self.groups, self.votes), prev_lkl)
     
  def test_cond_mean_var_xi(self):
    prev_lkl = likelihood(self.groups, self.votes)
    var = self.groups['xi'].iter_variables().next()
    var.cond_var = None
    var.var_dot = None
    mean_res, var_res = var.get_cond_mean_and_var(self.groups, self.votes_dict)
    variance = 1.0 / (2.0 / self.groups['xi'].var_H.value +
        1.0 / self.groups['xi'].var_param.value)
    mean = variance * ((var.get_rest_value(self.groups, self.votes[0]) + \
        var.get_rest_value(self.groups, self.votes[1])) / \
        self.groups['xi'].var_H.value + \
        self.groups['xi'].weight_param.value.T.dot(var.features)[0,0] / \
        self.groups['xi'].var_param.value)
    self.assertAlmostEqual(mean_res, mean)
    self.assertAlmostEqual(var_res, variance)
    var.update(float(mean_res))
    self.assertGreaterEqual(likelihood(self.groups, self.votes), prev_lkl)

  def test_e_step(self):
    """ Because of Gibbs Sampling, there is no guarantee that EM iterations will
        always improve the likelihood. But we assume that the first iteration
        will in most cases.
    """
    for _ in xrange(2):
      vote = self.votes[0]
      beta_0 = self.groups['beta'].get_instance(vote).value
      alpha_0 = self.groups['alpha'].iter_variables().next().value
      xi_0 = self.groups['xi'].iter_variables().next().value
      gamma_0 = self.groups['gamma'].iter_variables().next().value
      lambd_0 = self.groups['lambda'].iter_variables().next().value
      uv_0 = self.groups['u'].iter_variables().next().value.T \
        .dot(self.groups['v'].get_instance(vote).value)[0,0]
      old_likel = likelihood(self.groups, self.votes)
      old_pred = self.groups['beta'].get_instance(vote).value + \
          self.groups['alpha'].iter_variables().next().value + \
          self.groups['xi'].iter_variables().next().value + \
          self.groups['gamma'].iter_variables().next().value + \
          self.groups['lambda'].iter_variables().next().value + \
          self.groups['u'].iter_variables().next().value.T \
          .dot(self.groups['v'].get_instance(vote).value)[0,0]
      em.perform_e_step(self.groups, self.d_groups, self.votes_dict, self.d_vars, 10)
      pred_0 = self.groups['beta'].get_instance(vote).value + \
          self.groups['alpha'].iter_variables().next().value + \
          self.groups['xi'].iter_variables().next().value + \
          self.groups['gamma'].iter_variables().next().value + \
          self.groups['lambda'].iter_variables().next().value + \
          self.groups['u'].iter_variables().next().value.T \
          .dot(self.groups['v'].get_instance(vote).value)[0,0]
      beta = self.groups['beta'].get_instance(vote)
      alpha = self.groups['alpha'].iter_variables().next()
      xi = self.groups['xi'].iter_variables().next()
      gamma = self.groups['gamma'].iter_variables().next()
      lambd = self.groups['lambda'].iter_variables().next()
      uv = self.groups['u'].iter_variables().next().value.T \
        .dot(self.groups['v'].get_instance(vote).value)[0,0]
      print vote['vote'], old_pred, pred_0
      print alpha_0, alpha.value
      print beta_0, beta.value
      print xi_0, xi.value
      print uv_0, uv
      print gamma_0, gamma.value
      print lambd_0, lambd.value
     # self.assertGreaterEqual(likelihood(self.groups, self.votes), old_likel)
     # self.assertGreaterEqual((vote['vote'] - old_pred) ** 2, (vote['vote'] -
     #     pred_0) ** 2)
      u = self.groups['u'].iter_variables().next()
      v = self.groups['v'].get_instance(vote)
      g_0 = self.groups['beta'].weight_param.value
      d_0 = self.groups['alpha'].weight_param.value
      b_0 = self.groups['xi'].weight_param.value
      r_0 = self.groups['gamma'].weight_param.value
      h_0 = self.groups['lambda'].weight_param.value
      W_0 = self.groups['u'].weight_param.value
      V_0 = self.groups['v'].weight_param.value
      em.perform_m_step(self.groups, self.votes_dict)
      g_n = self.groups['beta'].weight_param.value
      d_n = self.groups['alpha'].weight_param.value
      b_n = self.groups['xi'].weight_param.value
      r_n = self.groups['gamma'].weight_param.value
      h_n = self.groups['lambda'].weight_param.value
      W_n = self.groups['u'].weight_param.value
      V_n = self.groups['v'].weight_param.value
      print beta.value, g_0.T.dot(beta.features)[0,0], g_n.T.dot(beta.features)[0,0]
      print alpha.value, d_0.T.dot(alpha.features)[0,0], d_n.T.dot(alpha.features)[0,0]
      print xi.value, b_0.T.dot(xi.features)[0,0], b_n.T.dot(xi.features)[0,0]
      print gamma.value, r_0.T.dot(gamma.features)[0,0], \
          aux.sigmoid(r_n.T.dot(gamma.features)[0,0])
      print lambd.value, h_0.T.dot(lambd.features)[0,0], \
          aux.sigmoid(h_n.T.dot(lambd.features)[0,0])
      print u.value
      print W_0.dot(u.features)
      print W_n.dot(u.features)
      print v.value
      print V_0.dot(v.features)
      print V_n.dot(v.features)
      self.assertGreaterEqual(abs(beta.value - g_0.T.dot(beta.features)[0,0]), \
          abs(beta.value - g_n.T.dot(beta.features)[0,0]))
      self.assertGreaterEqual(abs(alpha.value - d_0.T.dot(alpha.features)[0,0]), \
          abs(alpha.value - d_n.T.dot(alpha.features)[0,0]))
      self.assertGreaterEqual(abs(xi.value - b_0.T.dot(xi.features)[0,0]), \
          abs(xi.value - b_n.T.dot(xi.features)[0,0]))
      self.assertGreaterEqual(abs(gamma.value - r_0.T.dot(gamma.features)[0,0]), \
          abs(gamma.value - aux.sigmoid(r_n.T.dot(gamma.features)[0,0])))
      new_likelihood = likelihood(self.groups, self.votes)
      self.groups['gamma'].weight_param.value = r_0
      self.assertGreaterEqual(new_likelihood, likelihood(self.groups, self.votes))
      self.assertGreaterEqual(abs(lambd.value - h_0.T.dot(lambd.features)[0,0]), \
          abs(lambd.value - aux.sigmoid(h_n.T.dot(lambd.features)[0,0])))
      self.groups['gamma'].weight_param.value = r_n
      self.groups['lambda'].weight_param.value = h_0
      self.assertGreaterEqual(new_likelihood, likelihood(self.groups, self.votes))
      self.groups['lambda'].weight_param.value = h_n
      self.assertGreaterEqual(sum(absolute(u.value - W_0.dot(u.features))), \
          sum(absolute(u.value - W_n.dot(u.features))))
      self.assertGreaterEqual(sum(absolute(v.value - V_0.dot(v.features))), \
          sum(absolute(v.value - V_n.dot(v.features))))


if __name__ == '__main__':
  main()
