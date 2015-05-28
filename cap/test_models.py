import unittest
from numpy import array, reshape
from numpy import testing as ntest

import cap.models2 as models


class ScalarValueTestCase(unittest.TestCase):
  """ Test case for a generic scalar value object. """

  def setUp(self):
    self.value = models.Value("scalar", 1)

  def test_init(self):
    self.assertEquals(self.value.name, "scalar")
    self.assertTrue(self.value.value > 0)
    self.assertLess(round(self.value.value, 6), 1) 

  def test_init_error(self):
    self.assertRaises(TypeError, lambda: models.Value("scalar", 2))
    self.assertRaises(TypeError, lambda: models.Value("scalar", 0))
    self.assertRaises(TypeError, lambda: models.Value("scalar", -1))
    self.assertRaises(TypeError, lambda: models.Value("scalar", 713))
    self.assertRaises(TypeError, lambda: models.Value("scalar", -87))

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
  """ Test case for generic vector value object. """

  def setUp(self):
    self.value = models.Value("array", (7, 1))

  def test_init(self):
    self.assertEquals(self.value.name, "array")
    for i in xrange(0, 7):
      self.assertTrue(self.value.value[i,0] > 0)
      self.assertLess(round(self.value.value[i,0], 6), 1) 

  def test_init_error(self):
    self.assertRaises(TypeError, lambda: models.Value("array", []))
    self.assertRaises(TypeError, lambda: models.Value("array", ()))
    self.assertRaises(TypeError, lambda: models.Value("array", (0,1)))
    self.assertRaises(TypeError, lambda: models.Value("array", 0))
    self.assertRaises(TypeError, lambda: models.Value("array", (1,3,4)))
    self.assertRaises(TypeError, lambda: models.Value("array", [1,3,4]))

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
  """ Test case for generic matrix value object. """

  def setUp(self):
    self.value = models.Value("array", (3, 3))

  def test_init(self):
    self.assertEquals(self.value.name, "array")
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
  """ Test case for a scalar variance parameter. """

  def setUp(self):
    self.review = [
    self.votes = [{'review': , 'reviewer': , 'voter': , 'truth'}]
    # put random features values: voter, review, author


if __name__ == '__main__':
  unittest.main()
