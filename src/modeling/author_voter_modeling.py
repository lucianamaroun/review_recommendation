""" Author-voter interaction modeling module
    ----------------------------------------

    Models author-voter relation set of features comparative features, both
    regarding preference similarity and interaction in trust network.

    Usage:
      Used only as a module, not directly callable.
"""


from math import log, isnan

import numpy as np
from numpy import identity
from numpy.linalg import pinv, inv
from networkx import adjacency_matrix, weakly_connected_component_subgraphs
from scipy.stats import pearsonr
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.linalg import lu

from src.util.aux import cosine, vectorize

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
  return float(len(inters)) / len(union) if len(union) > 0 else 0.0


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


""" Computes Katz matrix. For each pair of uesrs, it is defined as the sum 
    over simple path lengths of the number of paths with the respective 
    length times beta. Directed paths are considered.

    Args:
      trusts: networkx DiGraph object with trust network.

    Returns:
      A matrix indexed by node ids with katz index of each pair. 
"""      
def get_katz_matrix(trusts):
 # n = len(trusts.nodes())
 # A = adjacency_matrix(trusts, sorted(trusts.nodes()))
 # katz = pinv(identity(n) - _BETA * A) - identity(n)
 # print "Ended Katz"
 # import pickle
 # pickle.dump(katz, open("pkl/katz.pkl", "w"))
 # print "Ended Dump"
 # import sys
 # sys.exit()
  katz = pickle.load(open("pkl/katz.pkl", "r"))
  nodes_index = {}
  for i, node in enumerate(sorted(trusts.nodes())):
    nodes_index[node] = i
  return katz, nodes_index


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
  author_ratings, voter_ratings = vectorize(author['ratings'],
      voter['ratings'])
  features['common_rated'] = len(author_rated.intersection(voter_rated))
  features['jacc_rated'] = jaccard(author_rated, voter_rated)
  features['cos_ratings'] = cosine(author_ratings, voter_ratings)
  pearson = pearsonr(author_ratings, voter_ratings)[0]  
  features['pear_ratings'] = pearson if not isnan(pearson) else 0.0 
  features['diff_avg_ratings'] = author['avg_rating'] - voter['avg_rating']
  if not author['ratings'].values() or not voter['ratings'].values():
    features['diff_max_ratings'] = -1
    features['diff_min_ratings'] = -1
  else:
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
def calculate_connection_strength(author, voter, trusts, katz, nodes_index):
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
  features['katz'] = 0
  if a_id in nodes_index and v_id in nodes_index:
    a_index = nodes_index[a_id]
    v_index = nodes_indes[v_id]
    features['katz'] = katz[v_index,a_index]
  return features


""" Models users similarities using votes in the training set. 

    Args:
      train: list of votes in the training set.
      users: dictionary of user models.

    Returns:
      A dictionary of similarity features indexed by a pair of user ids.
"""
def model_author_voter_similarity(train, users, similar):
  sim_features = {}
  for vote in train:
    author_id = vote['reviewer']
    voter_id = vote['voter']
    if (author_id, voter_id) in sim_features or author_id not in \
      similar[voter_id]:
      continue
    sim_features[(author_id, voter_id)] = \
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
  katz, nodes_index = None, None # get_katz_matrix(trusts) 
  for vote in train:
    author_id = vote['reviewer']
    voter_id = vote['voter']
    if (author_id, voter_id) in conn_features or voter_id not in trusts or \
      author_id not in trusts[voter_id]:
      continue
    conn_features[(author_id, voter_id)] = \
        calculate_connection_strength(users[author_id], users[voter_id], trusts,
        katz, nodes_index)

  return conn_features

