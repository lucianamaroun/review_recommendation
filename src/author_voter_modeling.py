""" Author-voter interaction modeling module
    ----------------------------------------

    Models author-voter relation set of features comparative features, both
    regarding preference similarity and interaction in trust network.

    Usage:
      Used only as a module, not directly callable.
"""


from math import log

import numpy as np
import networkx as nx
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr


_BETA = 0.005


""" Computes jaccard similarity between two sets.

    Args:
      set_a: first set of values (type has to be set).
      set_b: second set of values (type has to be set).

    Returns:
      A real value with the jaccard similarity coefficient.
"""
def jaccard(set_a, set_b):
  inters = set_a.intersection(set_b)
  union = set_a.union(set_b)
  return float(len(inters)) / len(union)


""" Computes Adamic-Adar index regarding commons trustors (in-degree). This
    scores sums the inverse of the log of common trustors degrees (both out and
    in).

    Args:
      trusts: networkx DiGraph object with trust network.
      user_a: id of the first user (order is not relevant, though).
      user_b: id of the second user.

    Returns:
      A real values with the Adamic-Adar index.
"""
def adamic_adar_trustors(trusts, user_a, user_b):
  trustors_a = set(trusts.predecessors(user_a))
  trustors_b = set(trusts.predecessors(user_b))
  common = trustors_a.intersection(trustors_b)
  score = 0.0
  for user in common:
    score += 1.0 / log(trusts.in_degree(user) + trusts.out_degree(user))
  return score


""" Computes Adamic-Adar index regarding commons trustees (out-degree). This
    scores sums the inverse of the log of common trustees degrees (both out and
    in).

    Args:
      trusts: networkx DiGraph object with trust network.
      user_a: id of the first user (order is not relevant, though).
      user_b: id of the second user.

    Returns:
      A real values with the Adamic-Adar index.
"""
def adamic_adar_trustees(trusts, user_a, user_b):
  trustees_a = set(trusts.successors(user_a))
  trustees_b = set(trusts.successors(user_b))
  common = trustees_a.intersection(trustees_b)
  score = 0.0
  for user in common:
    score += 1.0 / log(trusts.in_degree(user) + trusts.out_degree(user))
  return score


""" Computes Katz index between two users. It is defined as the sum over simple
    path lengths of the number of path with the respective length time beta.
    Paths in both ways are considered.

    Args:
      trusts: networkx DiGraph object with trust network.
      user_a: id of the first user (order is not relevant, though).
      user_b: id of the second user.

    Returns:
      A real values with the Katz index.
"""      
def katz(trusts, user_a, user_b):
  paths_a = nx.all_simple_paths(trusts, user_a, user_b)
  paths_b = nx.all_simple_paths(trusts, user_b, user_a)
  path_lengths = [len(p) - 1 for p in paths_a] + [len(p) - 1 for p in paths_b]
  unique_lengths = set(path_lengths)
  score = 0.0
  for l in unique_lengths:
    score += _BETA * path_lengths.count(l)
  return score


""" Maps two dictionary of values into two vectors in a common space. Each
    unique key defines a dimension; if a key is absent, the value is interpreted
    as zero.

    Args:
      dict_a: dictionary containing the one set of values.
      dict_b: dictionary containing the another set of values.

    Returns:
      Two numpy arrays with the values in the common space. The number of
    dimensions is defined by the size of the union of dictionary keys.
"""
def obtain_vectors(dict_a, dict_b):
  dimensions = set(dict_a.keys()).union(set(dict_b.keys()))
  vec_a = np.zeros(len(dimensions))
  vec_b = np.zeros(len(dimensions))
  for dim_index, dim_name in enumerate(dimensions):
    vec_a[dim_index] = dict_a[dim_name] if dim_name in dict_a else 0
    vec_b[dim_index] = dict_b[dim_name] if dim_name in dict_b else 0
  return vec_a, vec_b


""" Calculates authoring similarity features between author and voter.

    Args:
      author: a dictionary with author containing individual user features.
      voter: a dictionary with voter containing individual user features.

    Returns:
      A dictionary with features represeting authoring similarity between author
    and voter.
"""
def calculate_authoring_similarity(author, voter):
  features = {}
  author_rated = set(author['ratings'].keys())
  voter_rated = set(voter['ratings'].keys())
  author_ratings, voter_ratings = obtain_vectors(author['ratings'],
      voter['ratings'])
  features['common_rated'] = len(author_rated.intersection(voter_rated))
  features['jacc_rated'] = jaccard(author_rated, voter_rated)
  features['cos_ratings'] = 1 - cosine(author_ratings, voter_ratings)
  features['pear_ratings'] = pearsonr(author_ratings, voter_ratings)[0]
  features['diff_avg_ratings'] = author['avg_rating'] - voter['avg_rating']
  features['diff_max_ratings'] = max(author['ratings'].values()) - \
      max(voter['ratings'].values())
  features['diff_min_ratings'] = min(author['ratings'].values()) - \
      min(voter['ratings'].values()) 
  return features


""" Calculates connection strength features between author and voter.

    Args:
      author: a dictionary with author containing individual user features.
      voter: a dictionary with voter containing individual user features.
    
    Returns:
      A dictionary with features represeting connection strength in trust
    netwoek between author and voter.
"""
def calculate_connection_strength(author, voter, trusts):
  features = {}
  a_id, v_id = author['id'], voter['id']
  author_trustees = set(trusts.successors(author['id']))
  voter_trustees = set(trusts.successors(voter['id']))
  author_trustors = set(trusts.predecessors(author['id']))
  voter_trustors = set(trusts.predecessors(voter['id']))
  features['jacc_trustees'] = jaccard(author_trustees, voter_trustees)
  features['jacc_trustors'] = jaccard(author_trustors, voter_trustors)
  features['adamic_adar_trustees'] = adamic_adar_trustees(trusts, a_id, v_id)
  features['adamic_adar_trustors'] = adamic_adar_trustors(trusts, a_id, v_id)
  features['katz'] = katz(trusts, a_id, v_id)
  return features


""" Models users similarities using votes in the training set. 

    Args:
      train: list of votes in the training set.
      users: dictionary of user models.

    Returns:
      A dictionary of similarity features indexed by a pair of user ids.
"""
def model_author_voter_similarity(train, users):
  sim_features = {}
  for vote in train:
    author_id = vote['reviewer']
    voter_id = vote['voter']
    if (voter_id, author_id) in sim_features or author_id not in \
        similar[voter_id]:
      continue
    sim_features[(voter_id, author_id)] = \
        calculate_authoring_similarity(users[author_id], users[voter_id])
  return sim_features


""" Models users connection strength using votes in the training set. 

    Args:
      train: list of votes in the training set.
      users: dictionary of user models.

    Returns:
      A dictionary of connection features indexed by a pair of user ids.
"""
def model_author_voter_connection(train, users, trusts):
  conn_features = {}
  for vote in train:
    author_id = vote['reviewer']
    voter_id = vote['voter']
    if (voter_id, author_id) in conn_features or author_id not in \
      trusts[voter_id]:
      continue
    conn_features[(voter_id, author_id)] = \
        calculate_connection_strength(users[author_id], users[voter_id], trusts)

  return conn_features

