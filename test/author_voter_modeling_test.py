from math import log, sqrt

import unittest
import numpy as np
import networkx as nx

import src.author_voter_modeling as av

class IsolatedJaccardTestCase(unittest.TestCase):
""" Test case for jaccard coefficient calculation between two sets. """

  def setUp(self):
    self.set_a = set([1, 3, 4, 6])
    self.set_b = set([1, 2, 4, 5, 7])
    self.inters = set([1, 4])
    self.union = set([1, 2, 3, 4, 5, 6, 7])

  def test_jaccard(self):
    score = float(len(self.inters)) / len(self.union)
    self.assertEqual(av.jaccard(self.set_a, self.set_b), score)


class IsolatedVectorsTestCase(unittest.TestCase):
""" Test case for obtaining vectors from dictionaries. Each key indicates a new
    dimension and an undefined one receives value 0. """
  def setUp(self):
    self.dict_a = {1: 1, 2: 2, 3: 4}
    self.dict_b = {0: -1, 2: 3, 4: 2}
    self.res_a = [0, 1, 2, 4, 0]
    self.res_b = [-1, 0, 3, 0, 2]

  def test_obtain_vectors(self):
    vec_a, vec_b = av.obtain_vectors(self.dict_a, self.dict_b)
    self.assertListEqual(vec_a.tolist(), self.res_a)
    self.assertListEqual(vec_b.tolist(), self.res_b)


class SimpleTestCase(unittest.TestCase):
""" Simple test case of author and voter, with all dependant functions
    considered. """
  def setUp(self):
    self.graph = nx.DiGraph()
    self.graph.add_edges_from([(3, 1), (4, 1), (1, 5), (1, 6), (1, 7), (2, 6),
        (2, 7), (2, 8), (4, 2), (9, 2), (7, 2), (6, 1), (5, 7)])
    self.author = {'id': 1, 'ratings': {1: 1, 2: 2, 4: 5, 5: 2},
        'avg_rating': 2.5}
    self.voter = {'id': 2, 'ratings': {2: 4, 5: 2, 6: 0}, 'avg_rating': 2}

  def test_jaccard_trustees(self):
    set_a, set_b = set([5, 6, 7]), set([6, 7, 8])
    inter = [6, 7]
    union = [5, 6, 7, 8]
    score = float(len(inter)) / len(union)
    self.assertEqual(av.jaccard(set_a, set_b), score)

  def test_jaccard_trustors(self):
    set_a, set_b = set([3, 4, 6]), set([4, 7, 9]) 
    inter = [4]
    union = [3, 4, 6, 7, 9]
    score = float(len(inter)) / len(union)
    self.assertEqual(av.jaccard(set_a, set_b), score)

  def test_obtain_vectors(self):
    vec_a, vec_b = av.obtain_vectors(self.author['ratings'],
        self.voter['ratings'])
    self.assertListEqual(vec_a.tolist(), [1, 2, 5, 2, 0])
    self.assertListEqual(vec_b.tolist(), [0, 4, 0, 2, 0])

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

  def test_connection_strength(self):
    features = av.calculate_connection_strength(self.author, self.voter,
        self.graph)
    answer = {}
    inter_trustees = [6, 7]
    union_trustees = [5, 6, 7, 8]
    answer['jacc_trustees'] = float(len(inter_trustees)) / len(union_trustees)
    inter_trustors = [4]
    union_trustors = [3, 4, 6, 7, 9]
    answer['jacc_trustors'] = float(len(inter_trustors)) / len(union_trustors)
    nodes = [6, 7]
    answer['adamic_adar_trustees'] = sum([1.0 / log(self.graph.in_degree(u) + 
        self.graph.out_degree(u)) for u in nodes])
    nodes = [4]
    answer['adamic_adar_trustors'] = sum([1.0 / log(self.graph.in_degree(u) + 
        self.graph.out_degree(u)) for u in nodes])
    ls = [2, 3]
    ls_count = {2: 2, 3: 1}
    answer['katz'] = sum([0.005 * ls_count[l] for l in ls]) 
    self.assertDictEqual(features, answer)

  def test_authoring_similarity(self):
    features = av.calculate_authoring_similarity(self.author, self.voter)
    answer = {}
    inter = [2, 5]
    union = [1, 2, 4, 5, 6]
    answer['common_rated'] = 2
    answer['jacc_rated'] = float(len(inter)) / len(union)
    vec_a = [1, 2, 5, 2, 0]
    vec_b = [0, 4, 0, 2, 0]
    norm_a = sqrt(sum([a**2 for a in vec_a]))
    norm_b = sqrt(sum([b**2 for b in vec_b]))
    answer['cos_ratings'] = sum([a*b for a,b in zip(vec_a, vec_b)]) / \
        (norm_a * norm_b) 
    features['cos_ratings'] = round(features['cos_ratings'], 16)
    mean_a = np.mean(vec_a)
    mean_b = np.mean(vec_b)
    dp_a = sqrt(sum([(a-mean_a)**2 for a in vec_a]))
    dp_b = sqrt(sum([(b-mean_b)**2 for b in vec_b]))
    answer['pear_ratings'] = sum([(a-mean_a)*(b-mean_b) for a,b in 
        zip(vec_a, vec_b)]) / (dp_a * dp_b)
    answer['diff_avg_ratings'] = self.author['avg_rating'] - \
        self.voter['avg_rating']
    answer['diff_max_ratings'] = max(self.author['ratings'].values()) - \
        max(self.voter['ratings'].values())
    answer['diff_min_ratings'] = min(self.author['ratings'].values()) - \
        min(self.voter['ratings'].values())
    self.assertDictEqual(features, answer)


if __name__ == '__main__':
  unittest.main()
