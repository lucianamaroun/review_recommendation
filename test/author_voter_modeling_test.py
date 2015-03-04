import unittest
import numpy as np

from src.author_voter_modeling import obtain_vectors


class ObtainVectorTest(unittest.TestCase):

  def setUp(self):
    self.dict_a = {1: 1, 2: 2, 3: 4}
    self.dict_b = {0: -1, 2: 3, 4: 2}
    self.res_a = [0, 1, 2, 4, 0]
    self.res_b = [-1, 0, 3, 0, 2]

  def test_obtain_vectors(self):
    vec_a, vec_b = obtain_vectors(self.dict_a, self.dict_b)
    self.assertListEqual(vec_a.tolist(), self.res_a)
    self.assertListEqual(vec_b.tolist(), self.res_b)


if __name__ == '__main__':
  unittest.main()
