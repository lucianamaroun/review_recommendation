from math import log

import unittest
import numpy as np
import networkx as nx

import src.author_voter_modeling as av


class SimilarityMetrics(unittest.TestCase):

  def setUp(self):
    self.set_a = set([1, 3, 4, 6])
    self.set_b = set([1, 2, 4, 5, 7])
    self.inters = set([1, 4])
    self.union = set([1, 2, 3, 4, 5, 6, 7])

  def test_jaccard(self):
    score = float(len(self.inters)) / len(self.union)
    self.assertEqual(av.jaccard(self.set_a, self.set_b), score)


class VectorModelTest(unittest.TestCase):

  def setUp(self):
    self.dict_a = {1: 1, 2: 2, 3: 4}
    self.dict_b = {0: -1, 2: 3, 4: 2}
    self.res_a = [0, 1, 2, 4, 0]
    self.res_b = [-1, 0, 3, 0, 2]

  def test_obtain_vectors(self):
    vec_a, vec_b = av.obtain_vectors(self.dict_a, self.dict_b)
    self.assertListEqual(vec_a.tolist(), self.res_a)
    self.assertListEqual(vec_b.tolist(), self.res_b)


class NetworkMetricsTest(unittest.TestCase):

  def setUp(self):
    self.graph = nx.DiGraph()
    self.graph.add_edges_from([(3, 1), (4, 1), (1, 5), (1, 6), (1, 7), (2, 6),
        (2, 7), (2, 8), (4, 2), (9, 2), (7, 2), (6, 1), (5, 7)])

  def test_adamic_adar_trustees(self):
    nodes = [6, 7]
    score = sum([1.0 / log(self.graph.in_degree(u) + self.graph.out_degree(u))
      for u in nodes])
    self.assertEqual(av.adamic_adar_trustees(self.graph, 1, 2), score)

  def test_adamic_adar_trustees(self):
    nodes = [6, 7]
    score = sum([1.0 / log(self.graph.in_degree(u) + self.graph.out_degree(u))
      for u in nodes])
    self.assertEqual(av.adamic_adar_trustees(self.graph, 1, 2), score)

  def test_adamic_adar_trustors(self):
    nodes = [4]
    score = sum([1.0 / log(self.graph.in_degree(u) + self.graph.out_degree(u))
      for u in nodes])
    self.assertEqual(av.adamic_adar_trustors(self.graph, 1, 2), score)


  def test_katz(self):
    ls = [2, 3]
    ls_count = {2: 2, 3: 1}
    score = sum([0.005 * ls_count[l] for l in ls])
    self.assertEqual(av.katz(self.graph, 1, 2), score)

if __name__ == '__main__':
  unittest.main()
