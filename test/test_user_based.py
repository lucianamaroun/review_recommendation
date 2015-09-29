import unittest
from numpy import array, nan, isnan

from util.aux import cosine
from methods.cf.user_based import UserBasedModel

class TinyTestCase(unittest.TestCase):

  def setUp(self):
    self.votes = [
        {'review': 'r1', 'author': 'a1', 'voter': 'v1', 'vote': 4},
        {'review': 'r1', 'author': 'a1', 'voter': 'v2', 'vote': 2},
        {'review': 'r1', 'author': 'a1', 'voter': 'v4', 'vote': 5},
        {'review': 'r2', 'author': 'a2', 'voter': 'v2', 'vote': 4},
        {'review': 'r2', 'author': 'a2', 'voter': 'v3', 'vote': 4},
        {'review': 'r2', 'author': 'a2', 'voter': 'v4', 'vote': 5},
        {'review': 'r3', 'author': 'a3', 'voter': 'v1', 'vote': 5},
        {'review': 'r3', 'author': 'a3', 'voter': 'v3', 'vote': 4},
        {'review': 'r3', 'author': 'a3', 'voter': 'v4', 'vote': 5},
    ]
    self.test = [
        {'review': 'r1', 'author': 'a1', 'voter': 'v3', 'vote': 3},
        {'review': 'r2', 'author': 'a1', 'voter': 'v1', 'vote': 5},
        {'review': 'r3', 'author': 'a3', 'voter': 'v2', 'vote': 1},
        {'review': 'r4', 'author': 'a2', 'voter': 'v3', 'vote': 4},
        {'review': 'r1', 'author': 'a1', 'voter': 'v5', 'vote': 4},
    ]
    self.pred = [
      # v3
      (cosine(array([0, 4, 4]), array([2, 4, 0])) * 2 + # v2 
      cosine(array([0, 4, 4]), array([5, 5, 5])) * 5) /  # v4
      (cosine(array([0, 4, 4]), array([2, 4, 0])) + 
      cosine(array([0, 4, 4]), array([5, 5, 5]))),
      # v1
      (cosine(array([4, 0, 5]), array([5, 5, 5])) * 5 + # v4
      cosine(array([4, 0, 5]), array([0, 4, 4])) * 4) / # v3
      (cosine(array([4, 0, 5]), array([5, 5, 5])) + 
      cosine(array([4, 0, 5]), array([0, 4, 4]))),  
      # v2
      (cosine(array([2, 4, 0]), array([5, 5, 5])) * 5 + # v4
      cosine(array([2, 4, 0]), array([0, 4, 4])) * 4) /  # v3
      (cosine(array([2, 4, 0]), array([5, 5, 5])) +
      cosine(array([2, 4, 0]), array([0, 4, 4]))),
      nan,
      nan
    ]
  
  def test_predict(self):
    mod = UserBasedModel(k=2)
    mod.fit(self.votes)
    pred = mod.predict(self.test)
    for i in xrange(3):
      self.assertAlmostEqual(self.pred[i], pred[i])
    for i in xrange(3, 5):
      self.assertTrue(isnan(pred[i]))
      
if __name__ == '__main__':
  unittest.main()
