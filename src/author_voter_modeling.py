""" Author-voter interaction modeling module
    ----------------------------------------

    Models author-voter relation set of features comparative features, both
    regarding preference similarity and interaction in trust network.

    Usage:
      Used only as a module, not directly callable.
"""


import numpy as np
import networkx as nx
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr


_BETA = 0.001


def jaccard(set_a, set_b):
  inters = set_a.intersection(set_b)
  union = set_a.union(set_b)
  return float(len(inters)) / len(union)


def adamic_adar_trustors(trusts, user_a, user_b):
  trustors_a = set(trusts.predecessors(user_a))
  trustors_b = set(trusts.predecessors(user_b))
  common = trustors_a.intersection(trustors_b)
  score = 0.0
  for user in common:
    scores += 1.0 / log(trusts.in_degree(user) + trusts.out_degree(user))
  return score


def adamic_adar_trustees(trusts, user_a, user_b):
  trustees_a = set(trusts.successors(user_a))
  trustees_b = set(trusts.successors(user_b))
  common = trustees_a.intersection(trustees_b)
  score = 0.0
  for user in common:
    scores += 1.0 / log(trusts.in_degree(user) + trusts.out_degree(user))
  return score


def katz(trusts, user_a, user_b):
  paths = nx.all_simple_paths(trusts, user_a, user_b)
  path_lengths = [len(p) for p in paths]
  unique_lenghts = set(path_lengths)
  score = 0.0
  for l in unique_lengths:
    score += _BETA * path_lengths.count(l)
  return score


def obtain_vectors(dict_a, dict_b):
  dimensions = set(dict_a.keys()).union(set(dict_b.keys()))
  vec_a = np.zeros(len(dimensions))
  vec_b = np.zeros(len(dimensions))
  for dim_index, dim_name in enumerate(dimensions):
    vec_a[dim_index] = dict_a[dim_name] if dim_name in dict_a else 0.0
    vec_b[dim_index] = dict_b[dim_name] if dim_name in dict_b else 0.0
  return vec_a, vec_b


def calculate_authoring_similarity(author, voter):
  features = {}
  author_rated = set(author['ratings'].keys()), set(voter['ratings'].keys())
  author_ratings, voter_ratings = obtain_vectors(dict_a, dict_b)
  features['common_rated'] = len(author_rated.intersection(voter_rated))
  features['jacc_rated'] = jaccard(author_rated, voter_rated)
  features['cos_ratings'] = 1 - cosine(author_ratings, voter_ratings)
  features['pear_ratings'] = pearsonr(author_ratings, voter_ratings)
  features['diff_avg_ratings'] = author['avg_rating'] - voter['avg_rating']
  features['diff_max_ratings'] = max(author_ratings) - max(voter_ratings)
  features['diff_min_ratings'] = min(author_ratings) - min(voter_ratings) 
  return features


def calculate_connection_strength(author, voter, trusts):
  features = {}
  author_trustees = set(trusts.successors(author['id']))
  voter_trustees = set(trusts.successors(voter['id']))
  author_trustors = set(trusts.predecessors(author['id']))
  voter_trustors = set(trusts.predecessors(voter['id']))
  features['jacc_trustees'] = jaccard(author_trustees, voter_trustees)
  features['jacc_trustors'] = jaccard(author_trustors, voter_trustors)
  features['adamic_adar_trustees'] = adamic_adar_trustees(trusts, author, voter)
  features['adamic_adar_trustors'] = adamic_adar_trustors(trusts, author, voter)
  features['katz'] = katz(trusts, author, voter)

