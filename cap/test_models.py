import unittest
from numpy import array, reshape
from numpy import testing as ntest

from cap import models
from cap import const

class ScalarValueTestCase(unittest.TestCase):
  ''' Test case for a generic scalar value object. '''

  def setUp(self):
    self.value = models.Value('scalar', 1)

  def test_init(self):
    self.assertEquals(self.value.name, 'scalar')
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
    self.assertEquals(self.value.value, 542.)

  def test_update_error(self):
    self.assertRaises(TypeError, lambda: self.value.update([123., 323.])) 
    self.assertRaises(TypeError, lambda: self.value.update((123., 323., 764.))) 
    self.assertRaises(TypeError, lambda: self.value.update([]))
    self.assertRaises(TypeError, lambda: self.value.update(()))
    self.assertRaises(TypeError, lambda: self.value.update((34.,))) 


class VectorValueTestCase(unittest.TestCase):
  ''' Test case for generic vector value object. '''

  def setUp(self):
    self.value = models.Value('array', (7, 1))

  def test_init(self):
    self.assertEquals(self.value.name, 'array')
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


class MatrixValueTestCase(unittest.TestCase):
  ''' Test case for generic matrix value object. '''

  def setUp(self):
    self.value = models.Value('array', (3, 3))

  def test_init(self):
    self.assertEquals(self.value.name, 'array')
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


class TinyScenarioTestCase(unittest.TestCase):
  ''' Test case for a scalar variance parameter. '''

  def setUp(self):
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
      'v8': array([78, 100, 0.11, 3.1, 3.3, 3.9, 4.0, 4.2, 4.1])
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
    self.votes = [
        {'review': 'r1', 'reviewer': 'a1', 'voter': 'v1', 'truth': 4},
        {'review': 'r1', 'reviewer': 'a1', 'voter': 'v2', 'truth': 2},
        {'review': 'r1', 'reviewer': 'a1', 'voter': 'v3', 'truth': 3},
        {'review': 'r2', 'reviewer': 'a1', 'voter': 'v1', 'truth': 5},
        {'review': 'r2', 'reviewer': 'a1', 'voter': 'v4', 'truth': 5},
        {'review': 'r3', 'reviewer': 'a2', 'voter': 'v5', 'truth': 3},
        {'review': 'r4', 'reviewer': 'a2', 'voter': 'v6', 'truth': 5},
        {'review': 'r4', 'reviewer': 'a2', 'voter': 'v7', 'truth': 4},
        {'review': 'r4', 'reviewer': 'a2', 'voter': 'v3', 'truth': 4},
        {'review': 'r5', 'reviewer': 'a3', 'voter': 'v1', 'truth': 5},
        {'review': 'r5', 'reviewer': 'a3', 'voter': 'v4', 'truth': 5},
        {'review': 'r5', 'reviewer': 'a3', 'voter': 'v5', 'truth': 1}
    ]
    self.groups = {}
    self.var_H = models.Parameter('var_H', 1)

  def test_alpha_group_creation(self):
    self.groups['alpha'] = models.EntityScalarGroup('alpha', 'voter',
        models.Parameter('d', (9,1)), models.Parameter('var_alpha', 1),
        self.var_H)
    for e_id, e_feat in self.voters.iteritems():
      self.groups['alpha'].add_instance(e_id, e_feat)
      variable = self.groups['alpha'].variables[e_id]
      self.assertEquals(e_id, variable.entity_id) 
      ntest.assert_array_equal(reshape(e_feat, (e_feat.shape[0],1)),
          variable.features) 
    self.assertEquals(self.groups['alpha'].get_size(), len(self.voters))
    for variable in self.groups['alpha'].iter_instances():
      self.assertIsInstance(variable, models.EntityScalarVariable)
      self.assertEquals(variable.e_type, 'voter')

  def test_beta_group_creation(self):
    self.groups['beta'] = models.EntityScalarGroup('beta', 'review',
        models.Parameter('g', (17,1)), models.Parameter('var_beta', 1),
        self.var_H)
    for e_id, e_feat in self.reviews.iteritems():
      self.groups['beta'].add_instance(e_id, e_feat)
      variable = self.groups['beta'].variables[e_id]
      self.assertEquals(e_id, variable.entity_id) 
      ntest.assert_array_equal(reshape(e_feat, (e_feat.shape[0],1)),
          variable.features) 
    self.assertEquals(self.groups['beta'].get_size(), len(self.reviews))
    for variable in self.groups['beta'].iter_instances():
      self.assertIsInstance(variable, models.EntityScalarVariable)
      self.assertEquals(variable.e_type, 'review')

  def test_xi_group_creation(self):
    self.groups['xi'] = models.EntityScalarGroup('xi', 'reviewer',
        models.Parameter('b', (5,1)), models.Parameter('var_xi', 1),
        self.var_H)
    for e_id, e_feat in self.authors.iteritems():
      self.groups['xi'].add_instance(e_id, e_feat)
      variable = self.groups['xi'].variables[e_id]
      self.assertEquals(e_id, variable.entity_id) 
      ntest.assert_array_equal(reshape(e_feat, (e_feat.shape[0],1)),
          variable.features) 
    self.assertEquals(self.groups['xi'].get_size(), len(self.authors))
    for variable in self.groups['xi'].iter_instances():
      self.assertIsInstance(variable, models.EntityScalarVariable)
      self.assertEquals(variable.e_type, 'reviewer')

  def test_u_group_creation(self):
    self.groups['u'] = models.EntityArrayGroup('u', (const.K, 1), 'voter',
        models.Parameter('W', (const.K, 9)), models.Parameter('var_u', 1),
        self.var_H)
    for e_id, e_feat in self.voters.iteritems():
      self.groups['u'].add_instance(e_id, e_feat)
      variable = self.groups['u'].variables[e_id]
      self.assertEquals(e_id, variable.entity_id) 
      ntest.assert_array_equal(reshape(e_feat, (e_feat.shape[0],1)),
          variable.features) 
    self.assertEquals(self.groups['u'].get_size(), len(self.voters))
    for variable in self.groups['u'].iter_instances():
      self.assertIsInstance(variable, models.EntityArrayVariable)
      self.assertEquals(variable.e_type, 'voter')

  def test_v_group_creation(self):
    self.groups['v'] = models.EntityArrayGroup('v', (const.K, 1), 'review',
        models.Parameter('V', (const.K,17)), models.Parameter('var_v', 1),
        self.var_H)
    for e_id, e_feat in self.reviews.iteritems():
      self.groups['v'].add_instance(e_id, e_feat)
      variable = self.groups['v'].variables[e_id]
      self.assertEquals(e_id, variable.entity_id) 
      ntest.assert_array_equal(reshape(e_feat, (e_feat.shape[0],1)),
          variable.features) 
    self.assertEquals(self.groups['v'].get_size(), len(self.reviews))
    for variable in self.groups['v'].iter_instances():
      self.assertIsInstance(variable, models.EntityArrayVariable)
      self.assertEquals(variable.e_type, 'review')

  def test_gamma_group_creation(self):
    self.groups['gamma'] = models.InteractionScalarGroup('gamma', ('reviewer', 
        'voter'), models.Parameter('r', (7, 1)), 
        models.Parameter('var_gamma', 1), self.var_H)
    for e_id, e_feat in self.sim.iteritems():
      self.groups['gamma'].add_instance(e_id, e_feat)
      variable = self.groups['gamma'].variables[e_id]
      self.assertEquals(e_id, variable.entity_id) 
      ntest.assert_array_equal(reshape(e_feat, (e_feat.shape[0],1)),
          variable.features) 
    self.assertEquals(self.groups['gamma'].get_size(), len(self.sim))
    for variable in self.groups['gamma'].iter_instances():
      self.assertIsInstance(variable, models.InteractionScalarVariable)
      self.assertEquals(variable.e_type, ('reviewer', 'voter'))

  def test_lambda_group_creation(self):
    self.groups['lambda'] = models.InteractionScalarGroup('lambda', ('reviewer',
        'voter'), models.Parameter('h', (4,1)),
        models.Parameter('var_lambda', 1), self.var_H)
    for e_id, e_feat in self.conn.iteritems():
      self.groups['lambda'].add_instance(e_id, e_feat)
      variable = self.groups['lambda'].variables[e_id]
      self.assertEquals(e_id, variable.entity_id) 
      ntest.assert_array_equal(reshape(e_feat, (e_feat.shape[0],1)),
          variable.features) 
    self.assertEquals(self.groups['lambda'].get_size(), len(self.conn))
    for variable in self.groups['lambda'].iter_instances():
      self.assertIsInstance(variable, models.InteractionScalarVariable)
      self.assertEquals(variable.e_type, ('reviewer', 'voter'))

if __name__ == '__main__':
  unittest.main()
