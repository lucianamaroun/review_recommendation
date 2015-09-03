import unittest

from src.util.evaluation import calculate_dcg, calculate_ndcg

class RankTestCase(unittest.TestCase):

  def setUp(self):
    self.pred = [3, 3, 2, 2, 1, 0]
    self.truth = [3, 2, 3, 0, 1, 2]

  def test_dcg(self):
    self.assertAlmostEqual(calculate_dcg(self.pred, self.truth, 6), 13.84826363)

  def test_ndcf(self):
    self.assertAlmostEqual(calculate_ndcg(self.pred, self.truth, 6), 0.94881075)

if __name__ == '__main__':
  unittest.main()
