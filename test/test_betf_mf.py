from unittest import Testcase, main
from numpy import nan, empty

from methods.betf.mf import MF_Model, K


class SimpleTestCase(TestCase):

  def setUp(self):
    self.votes = [
        {'review': 'r1', 'author': 'a1', 'voter': 'v1', 'vote': 4},
        {'review': 'r1', 'author': 'a1', 'voter': 'v2', 'vote': 2},
        {'review': 'r1', 'author': 'a1', 'voter': 'v3', 'vote': 3},
        {'review': 'r2', 'author': 'a1', 'voter': 'v1', 'vote': 5},
        {'review': 'r2', 'author': 'a1', 'voter': 'v4', 'vote': 5},
        {'review': 'r3', 'author': 'a2', 'voter': 'v5', 'vote': 3},
        {'review': 'r4', 'author': 'a2', 'voter': 'v6', 'vote': 5},
        {'review': 'r4', 'author': 'a2', 'voter': 'v7', 'vote': 4},
        {'review': 'r4', 'author': 'a2', 'voter': 'v3', 'vote': 4},
        {'review': 'r5', 'author': 'a3', 'voter': 'v1', 'vote': 5},
        {'review': 'r5', 'author': 'a3', 'voter': 'v4', 'vote': 5},
        {'review': 'r5', 'author': 'a3', 'voter': 'v5', 'vote': 1}
    ]
    self.u_size = 7
    self.r_size = 5
    self.u_set = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']
    self.r_set = ['r1', 'r2', 'r3', 'r4', 'r5']
    self.model = None 

  def test_initialize(self):
    self.model = MF_Model()
    self.model.initialize_matrices(self.votes)
    for i in xrange(self.u_size):
      for j in xrange(K):
        self.assertTrue(self.model.U[i,j] >= 0 and self.model.U[i,j] < 1)
    for i in xrange(self.r_size):
      for j in xrange(K):
        self.assertTrue(self.model.R[i,j] >= 0 and self.model.R[i,j] < 1)
    for u_id in self.u_set:
      self.assertTrue(self.model.user_map[u_id] >= 0 and \
          self.model.user_map[u_id] < self.u_size)
    for r_id in self.r_set:
      self.assertTrue(self.model.review_map[r_id] >= 0 and \
          self.model.review_map[r_id] < self.r_size)
    self.assertEqual(len(set([self.model.user_map[u_id] for u_id in self.u_set])),
        self.u_size)
    self.assertEqual(len(set([self.model.review_map[r_id] for r_id in self.r_set])),
        self.r_size)
  
  def test_fit(self):
    self.model = MF_Model()
    self.model.fit(self.votes)
    U, R = self.model.U, self.model.R
    for vote in self.votes:
      u = self.model.user_map[vote['voter']]
      r = self.model.review_map[vote['review']]
      self.assertAlmostEqual(vote['vote'], U[u,:].dot(R[r,:].T), 1)
  

if __name__ == '__main__':
  main()
