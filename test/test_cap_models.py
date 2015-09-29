""" Test of Models
    --------------

    Test variables and parameters models with basic test cases.

    Usage:
    $ python -m test.test_models
"""


from unittest import TestCase, main
from numpy import array, reshape, identity, vstack, diagonal, zeros
from numpy import testing as ntest
from numpy.linalg import pinv, lstsq
from random import random

from algorithms.cap import models, const
from util import aux


class ScalarValueTestCase(TestCase):
  """ Test case for a generic scalar value object. """

  def setUp(self):
    const.K = 10
    self.value = models.Value('scalar', 1)

  def test_init(self):
    self.assertEqual(self.value.name, 'scalar')
    self.assertTrue(self.value.value > 0)
    self.assertLess(round(self.value.value, 6), 1) 

  def test_init_error(self):
    self.assertRaises(TypeError, lambda: models.Value('scalar', 2))
    self.assertRaises(TypeError, lambda: models.Value('scalar', 0))
    self.assertRaises(TypeError, lambda: models.Value('scalar', -1))
    self.assertRaises(TypeError, lambda: models.Value('scalar', 713))
    self.assertRaises(TypeError, lambda: models.Value('scalar', -87))

  def test_update(self):
    self.value.update(542.)
    self.assertEqual(self.value.value, 542.)

  def test_update_error(self):
    self.assertRaises(TypeError, lambda: self.value.update([123., 323.])) 
    self.assertRaises(TypeError, lambda: self.value.update((123., 323., 764.))) 
    self.assertRaises(TypeError, lambda: self.value.update([]))
    self.assertRaises(TypeError, lambda: self.value.update(()))
    self.assertRaises(TypeError, lambda: self.value.update((34.,))) 


class VectorValueTestCase(TestCase):
  """ Test case for generic vector value object. """

  def setUp(self):
    const.K = 10
    self.value = models.Value('array', (7, 1))

  def test_init(self):
    self.assertEqual(self.value.name, 'array')
    for i in xrange(0, 7):
      self.assertTrue(self.value.value[i,0] > 0)
      self.assertLess(round(self.value.value[i,0], 6), 1) 

  def test_init_error(self):
    self.assertRaises(TypeError, lambda: models.Value('array', []))
    self.assertRaises(TypeError, lambda: models.Value('array', ()))
    self.assertRaises(TypeError, lambda: models.Value('array', (0,1)))
    self.assertRaises(TypeError, lambda: models.Value('array', 0))
    self.assertRaises(TypeError, lambda: models.Value('array', (1,3,4)))
    self.assertRaises(TypeError, lambda: models.Value('array', [1,3,4]))

  def test_update(self):
    new_value = reshape(array([1, 2, 3, 4, 5, 6, 7]), (7, 1))
    self.value.update(new_value)
    ntest.assert_array_equal(self.value.value, new_value)

  def test_update_error(self):
    self.assertRaises(TypeError, lambda: self.value.update([123., 323.])) 
    self.assertRaises(TypeError, lambda: self.value.update((123., 323., 764.))) 
    self.assertRaises(TypeError, lambda: self.value.update([]))
    self.assertRaises(TypeError, lambda: self.value.update(()))
    self.assertRaises(TypeError, lambda: self.value.update((34.,))) 
    self.assertRaises(TypeError, lambda: self.value.update(3)) 


class MatrixValueTestCase(TestCase):
  """ Test case for generic matrix value object. """

  def setUp(self):
    const.K = 10
    self.value = models.Value('array', (3, 3))

  def test_init(self):
    self.assertEqual(self.value.name, 'array')
    for i in xrange(0, 3):
      for j in xrange(0, 3):
        self.assertTrue(self.value.value[i,j] > 0)
        self.assertLess(round(self.value.value[i,j], 6), 1) 

  def test_update(self):
    new_value = reshape(array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), (3, 3))
    self.value.update(new_value)
    ntest.assert_array_equal(self.value.value, new_value)

  def test_update_error(self):
    self.assertRaises(TypeError, lambda: self.value.update([123., 323.])) 
    self.assertRaises(TypeError, lambda: self.value.update((123., 323., 764.))) 
    self.assertRaises(TypeError, lambda: self.value.update([]))
    self.assertRaises(TypeError, lambda: self.value.update(()))
    self.assertRaises(TypeError, lambda: self.value.update((34.,))) 
    self.assertRaises(TypeError, lambda: self.value.update(3)) 
    self.assertRaises(TypeError, lambda: self.value.update(array([[2,2], [2,2],
        [3,3]]))) 


class SmallScenarioTestCase(TestCase):
  """ Test case for a scalar variance parameter. """

  def setUp(self):
    const.K = 10
    self.reviews = {
        'r1': array([50, 3, 0.5, 16.67, 1.0, 0.2, 0.3, 0.2, 0.6, 0.3, 0.01,
            0.003, 0.0, 0.02, 0.123, 0.4, 0.3]),
        'r2': array([100, 12, 0.62, 8.33, 0.9, 0.1, 0.4, 0.5, 0.4, 0.2, 0.0,
            0.0, 0.0, 0.2, 0.01, 0.2, 0.7]),
        'r3': array([80, 14, 0.4, 5.71, 0.8, 0.4, 0.5, 0.0, 0.1, 0.0, 0.0, 0.0,
            0.0, 0.01, 0.412, 0.3, 0.1]),
        'r4': array([150, 20, 0.76, 7.5, 1.0, 0.2, 0.4, 0.1, 0.1, 0.1, 0.0, 0.1,
            0.0, 0.01, 0.089, 0.5, 0.2]),
        'r5': array([200, 50, 0.3, 4, 0.2, 0.1, 0.2, 0.0, 0.4, 0.1, 0.1, 0.0,
            0.01, 0.04, 0.567, 0.68, 0.12])
    }
    self.authors = {
      'a1': array([23, 3.7, 10, 4, 0.5]),
      'a2': array([3, 2.3, 3, 20, 0.01]),
      'a3': array([402, 4.2, 170, 23, 0.9])
    }
    self.voters = {
      'v1': array([12, 17, 0.02, 3.8, 3.5, 3.9, 4.3, 4.5, 4.0]),
      'v2': array([0, 24, 0.001, 2.1, 1.8, 2.0, 3.6, 3.2, 3.3]),
      'v3': array([200, 47, 0.68, 3.4, 3.9, 3.7, 3.0, 4.1, 3.8]),
      'v4': array([50, 35, 0.52, 4.5, 3.8, 4.2, 4.8, 4.6, 4.3]),
      'v5': array([4, 87, 0.04, 3.2, 3.8, 2.8, 3.5, 4.1, 3.1]),
      'v6': array([16, 5, 0.1, 4.2, 4.5, 4.3, 3.8, 4.8, 4.9]),
      'v7': array([304, 509, 0.2, 4.7, 4.5, 4.9, 4.7, 4.7, 4.8]),
    }
    self.sim = {
      ('a1', 'v2'): array([3, 0.2, 0.4, 0.3, 1.6, 2, 3]),
      ('a1', 'v3'): array([5, 0.3, 0.6, 0.5, 0.3, 0, 1]),
      ('a2', 'v5'): array([20, 0.6, 0.7, 0.8, 0.2, 0, 0]),
      ('a2', 'v7'): array([13, 0.4, 0.42, 0.35, 1.7, 1, 2]),
      ('a3', 'v1'): array([7, 0.24, 0.29, 0.3, 0.4, 1, 0]),
      ('a3', 'v5'): array([18, 0.7, 0.8, 0.76, 1, 0, 3])
    }
    self.conn = {
      ('a1', 'v1'): array([0.02, 0.01, 0.2, 0.03]),
      ('a1', 'v3'): array([0.1, 0.23, 0.3, 0.13]), 
      ('a2', 'v6'): array([0.4, 0.1, 0.3, 0.01]),
      ('a2', 'v7'): array([0.5, 0.32, 0.6, 0.4]),
      ('a3', 'v5'): array([0.3, 0.2, 0.12, 0.19])
    }
    self.votes = {
        1: {'review': 'r1', 'author': 'a1', 'voter': 'v1', 'vote': 4},
        2: {'review': 'r1', 'author': 'a1', 'voter': 'v2', 'vote': 2},
        3: {'review': 'r1', 'author': 'a1', 'voter': 'v3', 'vote': 3},
        4: {'review': 'r2', 'author': 'a1', 'voter': 'v1', 'vote': 5},
        5: {'review': 'r2', 'author': 'a1', 'voter': 'v4', 'vote': 5},
        6: {'review': 'r3', 'author': 'a2', 'voter': 'v5', 'vote': 3},
        7: {'review': 'r4', 'author': 'a2', 'voter': 'v6', 'vote': 5},
        8: {'review': 'r4', 'author': 'a2', 'voter': 'v7', 'vote': 4},
        9: {'review': 'r4', 'author': 'a2', 'voter': 'v3', 'vote': 4},
        10: {'review': 'r5', 'author': 'a3', 'voter': 'v1', 'vote': 5},
        11: {'review': 'r5', 'author': 'a3', 'voter': 'v4', 'vote': 5},
        12: {'review': 'r5', 'author': 'a3', 'voter': 'v5', 'vote': 1}
    }
    self.groups = {}
    self.var_H = models.PredictionVarianceParameter('var_H')
    self._create_groups()
    self.groups['u'].set_pair_name('v')
    self.groups['v'].set_pair_name('u')
  
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
        models.EntityArrayParameter('V', (const.K, 17)),
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
        'voter'), models.InteractionScalarParameter('h', (4, 1)),
        models.ScalarVarianceParameter('var_lambda'), self.var_H)
    for e_id, e_feat in self.conn.iteritems():
      self.groups['lambda'].add_instance(e_id, e_feat)


  def test_alpha_group_creation(self):
    for e_id, e_feat in self.voters.iteritems():
      variable = self.groups['alpha'].variables[e_id]
      self.assertEqual(e_id, variable.entity_id) 
      ntest.assert_array_equal(reshape(e_feat, (e_feat.shape[0],1)),
          variable.features) 
    self.assertEqual(self.groups['alpha'].get_size(), len(self.voters))
    for variable in self.groups['alpha'].iter_variables():
      self.assertIsInstance(variable, models.EntityScalarVariable)
      self.assertEqual(variable.e_type, 'voter')

  def test_beta_group_creation(self):
    for e_id, e_feat in self.reviews.iteritems():
      variable = self.groups['beta'].variables[e_id]
      self.assertEqual(e_id, variable.entity_id) 
      ntest.assert_array_equal(reshape(e_feat, (e_feat.shape[0],1)),
          variable.features) 
    self.assertEqual(self.groups['beta'].get_size(), len(self.reviews))
    for variable in self.groups['beta'].iter_variables():
      self.assertIsInstance(variable, models.EntityScalarVariable)
      self.assertEqual(variable.e_type, 'review')

  def test_xi_group_creation(self):
    for e_id, e_feat in self.authors.iteritems():
      variable = self.groups['xi'].variables[e_id]
      self.assertEqual(e_id, variable.entity_id) 
      ntest.assert_array_equal(reshape(e_feat, (e_feat.shape[0],1)),
          variable.features) 
    self.assertEqual(self.groups['xi'].get_size(), len(self.authors))
    for variable in self.groups['xi'].iter_variables():
      self.assertIsInstance(variable, models.EntityScalarVariable)
      self.assertEqual(variable.e_type, 'author')

  def test_u_group_creation(self):
    for e_id, e_feat in self.voters.iteritems():
      variable = self.groups['u'].variables[e_id]
      self.assertEqual(e_id, variable.entity_id) 
      ntest.assert_array_equal(reshape(e_feat, (e_feat.shape[0],1)),
          variable.features) 
    self.assertEqual(self.groups['u'].get_size(), len(self.voters))
    for variable in self.groups['u'].iter_variables():
      self.assertIsInstance(variable, models.EntityArrayVariable)
      self.assertEqual(variable.e_type, 'voter')

  def test_v_group_creation(self):
    for e_id, e_feat in self.reviews.iteritems():
      variable = self.groups['v'].variables[e_id]
      self.assertEqual(e_id, variable.entity_id) 
      ntest.assert_array_equal(reshape(e_feat, (e_feat.shape[0],1)),
          variable.features) 
    self.assertEqual(self.groups['v'].get_size(), len(self.reviews))
    for variable in self.groups['v'].iter_variables():
      self.assertIsInstance(variable, models.EntityArrayVariable)
      self.assertEqual(variable.e_type, 'review')

  def test_gamma_group_creation(self):
    for e_id, e_feat in self.sim.iteritems():
      variable = self.groups['gamma'].variables[e_id]
      self.assertEqual(e_id, variable.entity_id) 
      ntest.assert_array_equal(reshape(e_feat, (e_feat.shape[0],1)),
          variable.features) 
    self.assertEqual(self.groups['gamma'].get_size(), len(self.sim))
    for variable in self.groups['gamma'].iter_variables():
      self.assertIsInstance(variable, models.InteractionScalarVariable)
      self.assertEqual(variable.e_type, ('author', 'voter'))

  def test_lambda_group_creation(self):
    for e_id, e_feat in self.conn.iteritems():
      variable = self.groups['lambda'].variables[e_id]
      self.assertEqual(e_id, variable.entity_id) 
      ntest.assert_array_equal(reshape(e_feat, (e_feat.shape[0],1)),
          variable.features) 
    self.assertEqual(self.groups['lambda'].get_size(), len(self.conn))
    for variable in self.groups['lambda'].iter_variables():
      self.assertIsInstance(variable, models.InteractionScalarVariable)
      self.assertEqual(variable.e_type, ('author', 'voter'))

  def test_empiric_stats(self):
    for g_id, group in self.groups.iteritems():
      for variable in group.iter_variables():
        if isinstance(variable, models.ScalarVariable):
          samples = [0.1*i for i in xrange(10)]
        else:
          samples = [reshape([0.1*j*i for j in
              xrange(variable.shape[0]*variable.shape[1])], variable.shape)
              for i in xrange(10)]
        for sample in samples:
          variable.add_sample(sample)
        variable.calculate_empiric_mean()
        variable.calculate_empiric_var()
        if isinstance(variable, models.ScalarVariable):
          self.assertAlmostEqual(variable.empiric_mean, 0.45)
          self.assertAlmostEqual(variable.empiric_var, 0.091666667)
        else:
          ntest.assert_allclose(variable.empiric_mean, reshape(array(
              [[0.0, 0.45, 0.9, 1.35, 1.8, 2.25, 2.7, 3.15, 3.6, 4.05]]),
              (10, 1)), rtol=1, atol=1e-7)
          ntest.assert_allclose(variable.empiric_var, reshape(array(
              [[0.0, 0.091666667, 0.366666667, 0.825, 1.466666667,
              2.291666667, 3.3, 4.491666667, 5.866666667, 7.425]]),
              (10, 1)), rtol=1, atol=1e-7)

  def test_get_instance_sample(self):
    for g_id, group in self.groups.iteritems():
      for variable in group.iter_variables():
        if type(variable.e_type) is tuple:
          vote = [v for v in self.votes.itervalues() if v[variable.e_type[0]] ==
              variable.entity_id[0] and v[variable.e_type[1]] ==
              variable.entity_id[1]][0]
        else:
          vote = [v for v in self.votes.itervalues() if v[variable.e_type] ==
              variable.entity_id][0]
        if isinstance(variable, models.ScalarVariable):
          self.assertEqual(group.get_instance(vote).get_last_sample(),
              variable.value)
          variable.add_sample(variable.value**2)
          self.assertEqual(group.get_instance(vote).get_last_sample(),
              variable.value**2)
        else:
          ntest.assert_array_equal(group.get_instance(vote).get_last_sample(),
              variable.value)
          variable.add_sample(variable.value**2)
          ntest.assert_array_equal(group.get_instance(vote).get_last_sample(),
              variable.value**2)

  def test_pair_name(self):
    self.assertEqual(self.groups['u'].pair_name, 'v')
    self.assertEqual(self.groups['v'].pair_name, 'u')

  def test_unpaired_get_rest_value(self):
    groups = self.groups
    variable = [v for v in groups['alpha'].iter_variables() if v.entity_id
        == 'v1' ][0]
    vote = self.votes.itervalues().next()
    rest = vote['vote'] - \
      [v for v in groups['beta'].iter_variables() if v.entity_id == 'r1'][0].value - \
      [v for v in groups['xi'].iter_variables() if v.entity_id == 'a1'][0].value - \
      [v for v in groups['u'].iter_variables() if v.entity_id =='v1'][0].value \
        .T.dot([v for v in groups['v'].iter_variables() if v.entity_id ==
        'r1'][0].value) - \
      [v for v in groups['lambda'].iter_variables() if v.entity_id == ('a1', 'v1')][0].value
    self.assertAlmostEqual(rest, variable.get_rest_value(groups, vote))

  def test_paired_get_rest_value(self):
    groups = self.groups
    variable = [v for v in groups['u'].iter_variables() if v.entity_id
        == 'v1' ][0]
    vote = self.votes.itervalues().next()
    rest = vote['vote'] - \
      [v for v in groups['alpha'].iter_variables() if v.entity_id == 'v1'][0].value - \
      [v for v in groups['beta'].iter_variables() if v.entity_id == 'r1'][0].value - \
      [v for v in groups['xi'].iter_variables() if v.entity_id == 'a1'][0].value - \
      [v for v in groups['lambda'].iter_variables() if v.entity_id == ('a1', 'v1')][0].value
    self.assertAlmostEqual(rest, variable.get_rest_value(groups, vote))

  def test_entity_scalar_variable_get_cond_mean_and_var(self):
    groups = self.groups
    group = groups['alpha']
    variable = [v for v in groups['alpha'].iter_variables() if v.entity_id
        == 'v1' ][0]
    related_votes = [
        {'review': 'r1', 'author': 'a1', 'voter': 'v1', 'vote': 4},
        {'review': 'r2', 'author': 'a1', 'voter': 'v1', 'vote': 5},
        {'review': 'r5', 'author': 'a3', 'voter': 'v1', 'vote': 5},
    ]
    true_var = 1 / (3 / self.var_H.value + 1 / group.var_param.value)
    rest_term = sum([variable.get_rest_value(groups, v) for v in
        related_votes]) / self.var_H.value
    dot_term = group.weight_param.value.T.dot(variable.features) / \
        group.var_param.value
    true_mean = true_var * (rest_term + dot_term[0,0])
    res_mean, res_var = variable.get_cond_mean_and_var(groups, self.votes)
    self.assertAlmostEqual(true_var, res_var)
    self.assertAlmostEqual(true_mean, res_mean)
  
  def test_entity_array_variable_get_cond_mean_and_var(self):
    groups = self.groups
    group = groups['u']
    variable = [v for v in groups['u'].iter_variables() if v.entity_id
        == 'v1' ][0]
    related_votes = [
        {'review': 'r1', 'author': 'a1', 'voter': 'v1', 'vote': 4},
        {'review': 'r2', 'author': 'a1', 'voter': 'v1', 'vote': 5},
        {'review': 'r5', 'author': 'a3', 'voter': 'v1', 'vote': 5},
    ]
    v_values = [[v for v in groups['v'].iter_variables() if v.entity_id ==
        vote['review']][0].value for vote in related_votes]
    v_sum = sum([v.dot(v.T) for v in v_values]) / self.var_H.value
    var_matrix = group.var_param.value * identity(const.K)
    inv_var = pinv(var_matrix)
    true_var = pinv(inv_var + v_sum)
    rest_term = sum([variable.get_rest_value(groups, related_votes[i]) * 
        v_values[i] for i in xrange(len(related_votes))]) / \
        group.var_H.value
    dot_term = inv_var.dot(group.weight_param.value) \
        .dot(variable.features)
    true_mean = true_var.dot(rest_term + dot_term)
    res_mean, res_var = variable.get_cond_mean_and_var(groups, self.votes)
    ntest.assert_allclose(true_var, res_var, rtol=1, atol=1e-7)
    ntest.assert_allclose(true_mean, res_mean, rtol=1, atol=1e-7)
  
  def test_interaction_scalar_variable_get_cond_mean_and_var(self):
    groups = self.groups
    group = groups['lambda']
    variable = [v for v in groups['lambda'].iter_variables() if v.entity_id
        == ('a1', 'v1')][0]
    related_votes = [
        {'review': 'r1', 'author': 'a1', 'voter': 'v1', 'vote': 4},
        {'review': 'r2', 'author': 'a1', 'voter': 'v1', 'vote': 5},
    ]
    true_var = 1 / (2 / self.var_H.value + 1 / group.var_param.value)
    rest_term = sum([variable.get_rest_value(groups, v) for v in
        related_votes]) / self.var_H.value
    dot_term = aux.sigmoid(group.weight_param.value.T.dot(variable.features)
        [0,0]) / group.var_param.value
    true_mean = true_var * (rest_term + dot_term)
    res_mean, res_var = variable.get_cond_mean_and_var(groups, self.votes)
    self.assertAlmostEqual(true_var, res_var)
    self.assertAlmostEqual(true_mean, res_mean)
 
  def test_scalar_optimize(self):
    groups = self.groups
    group = groups['alpha']
    matrix = None
    y = None 
    for variable in group.iter_variables():
      variable.num_samples = 10
      variable.samples = [random() for _ in xrange(10)] 
      variable.calculate_empiric_mean()
      variable.calculate_empiric_var()
      if matrix is None:
        size = variable.features.shape[0]
        matrix = variable.features.reshape(1, size)
        y = variable.empiric_mean
      else:
        size = variable.features.shape[0]
        matrix = vstack((matrix, variable.features.reshape(1, size)))
        y = vstack((y, variable.empiric_mean))
    #Scikit
    #reg = LinearRegression(fit_intercept=False)
    #reg.fit(matrix, y)
    #weight = reg.coef_.T
    #pred = reg.predict(matrix)
    #sse = sum(reshape(((y-pred)**2), y.shape[0]).tolist())
    #Numpy
    #weight, res = lstsq(matrix, y)[:2] # linear regression
    #sse = sum(res)
    weight = pinv(matrix.T.dot(matrix)).dot(matrix.T).dot(y)
    pred = matrix.dot(weight)
    sse = sum(reshape(((y-pred)**2), y.shape[0]).tolist())
    var_sum = sum([v.empiric_var for v in group.iter_variables()])
    var = (sse + var_sum) / group.get_size()
    group.weight_param.optimize(group)
    group.var_param.optimize(group)
    ntest.assert_allclose(weight, group.weight_param.value, rtol=1, atol=1e-7)
    self.assertAlmostEqual(var, group.var_param.value)

  def test_array_optimize(self):
    groups = self.groups
    group = groups['u']
    matrix = None
    y = None 
    for variable in group.iter_variables():
      variable.num_samples = 10
      variable.samples = [array([random()]*const.K).reshape(const.K, 1) for _ in xrange(10)] 
      variable.calculate_empiric_mean()
      variable.calculate_empiric_var()
      if matrix is None:
        size = variable.features.shape[0]
        matrix = variable.features.reshape(1, size)
        y = reshape(variable.empiric_mean, (1, const.K))
      else:
        size = variable.features.shape[0]
        matrix = vstack((matrix, variable.features.reshape(1, size)))
        y = vstack((y, variable.empiric_mean.reshape(1, variable.shape[0])))
    weight = pinv(matrix.T.dot(matrix)).dot(matrix.T).dot(y).T
    pred = weight.dot(matrix.T).T
    sse = sum(reshape(((y-pred)**2), -1).tolist())
    var_sum = sum([sum(v.empiric_var.reshape(-1).tolist()) for v in 
        group.iter_variables()])
    var = float(sse + var_sum) / (group.get_size() * const.K)
    group.weight_param.optimize(group)
    group.var_param.optimize(group)
    ntest.assert_allclose(weight, group.weight_param.value, rtol=1, atol=1e-7)
    self.assertAlmostEqual(var, group.var_param.value, 5)

  def test_interaction_get_der1(self):
    groups = self.groups
    group = groups['lambda']
    matrix = None
    y = None 
    der1 = 0
    param = group.weight_param
    for variable in group.iter_variables():
      variable.num_samples = 10
      variable.samples = [random() for _ in xrange(10)] 
      feat = variable.features
      dot = param.value.T.dot(feat)[0,0]
      for sample in variable.samples:
        der1 += (sample - aux.sigmoid(dot)) * aux.sigmoid_der1(dot) * feat
    der1 = 1/(group.var_param.value*10) * der1
    ntest.assert_allclose(der1, param.get_derivative_1(param.value, group), rtol=1, atol=1e-7)

  def test_interaction_get_der2(self):
    groups = self.groups
    group = groups['lambda']
    matrix = None
    y = None 
    der2 = 0
    param = group.weight_param
    for variable in group.iter_variables():
      variable.num_samples = 10
      variable.samples = [random() for _ in xrange(10)] 
      feat = variable.features
      dot = param.value.T.dot(feat)[0,0]
      for sample in variable.samples:
        der2 += (- aux.sigmoid_der1(dot) ** 2 + (sample - aux.sigmoid(dot)) * 
            aux.sigmoid_der2(dot)) * feat.dot(feat.T)
    der2 = 1/(group.var_param.value*10) * der2
    der2 = der2.reshape(group.weight_param.shape[0], group.weight_param.shape[0])
    ntest.assert_allclose(der2, param.get_derivative_2(param.value, group), rtol=1, atol=1e-7)

  def test_interaction_optimize(self):
    groups = self.groups
    group = groups['lambda']
    for variable in group.iter_variables():
      variable.num_samples = 10
      variable.samples = [random() for _ in xrange(10)]
    param = group.weight_param
    param.optimize(group)
    ntest.assert_allclose(zeros(param.value.shape), 
        param.get_derivative_1(param.value, group), rtol=1, atol=1e-3)


if __name__ == '__main__':
  main()
