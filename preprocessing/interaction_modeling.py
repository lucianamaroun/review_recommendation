""" Author-voter interaction modeling module
    ----------------------------------------

    Models author-voter relation set of features comparative features, both
    regarding preference similarity and interaction in trust network.

    Usage:
      Used only as a module, not directly callable.
"""


from math import log, isnan
from os.path import isfile

from numpy import identity, nan
from numpy.linalg import pinv
from networkx import adjacency_matrix
from pickle import load, dump
from scipy.stats import pearsonr

from util.aux import cosine, vectorize


_BETA = 0.005
_KATZ_PKL = 'out/pkl/katz.pkl'


def jaccard(set_a, set_b):
  """ Computes jaccard similarity between two sets.

      Args:
        set_a: first set of values (type has to be set).
        set_b: second set of values (type has to be set).

      Returns:
        A real value with the jaccard similarity coefficient.
  """
  inters = set_a.intersection(set_b)
  union = set_a.union(set_b)
  return float(len(inters)) / len(union) if len(union) > 0 else 0.0


def adamic_adar_trustors(trusts, user_a, user_b):
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
  trustors_a = set(trusts.predecessors(user_a))
  trustors_b = set(trusts.predecessors(user_b))
  common = trustors_a.intersection(trustors_b)
  score = 0.0
  for user in common:
    score += 1.0 / log(trusts.in_degree(user) + trusts.out_degree(user))
  return score


def adamic_adar_trustees(trusts, user_a, user_b):
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
  trustees_a = set(trusts.successors(user_a))
  trustees_b = set(trusts.successors(user_b))
  common = trustees_a.intersection(trustees_b)
  score = 0.0
  for user in common:
    try:
      score += 1.0 / log(trusts.in_degree(user) + trusts.out_degree(user))
    except:
      print trustees_a
      print trustees_b
      print common
      print '--- %s' % user
      print user in trusts[user_a]
      print user in trusts[user_b]
      print trusts.in_degree(user)
      print trusts.out_degree(user)
      import sys
      sys.exit()
  return score


def get_katz_matrix(trusts):
  """ Computes Katz matrix. For each pair of uesrs, it is defined as the sum 
      over simple path lengths of the number of paths with the respective 
      length times beta. Directed paths are considered.

      Args:
        trusts: networkx DiGraph object with trust network.

      Returns:
        A matrix indexed by node ids with katz index of each pair. 
  """      
  if isfile(_KATZ_PKL):
    katz = load(open(_KATZ_PKL, 'r'))
  else:
    n = len(trusts.nodes())
    A = adjacency_matrix(trusts, sorted(trusts.nodes()))
    katz = pinv(identity(n) - _BETA * A) - identity(n)
    dump(katz, open(_KATZ_PKL, 'w'))
  nodes_index = {}
  for i, node in enumerate(sorted(trusts.nodes())):
    nodes_index[node] = i
  return katz, nodes_index


def calculate_authoring_similarity(author, voter):
  """ Calculates authoring similarity features between author and voter.

      Args:
        author: a dictionary with author containing individual user features.
        voter: a dictionary with voter containing individual user features.

      Returns:
        A dictionary with features represeting authoring similarity between author
      and voter.
  """
  features = {}
  if not author and not voter:
    features['common_rated'] = nan 
    features['jacc_rated'] = nan 
    features['cos_ratings'] = nan 
    features['pear_ratings'] = nan 
    features['diff_avg_ratings'] = nan 
    features['diff_max_ratings'] = nan
    features['diff_min_ratings'] = nan
    return features 
  author_rated = set(author['ratings'].keys())
  voter_rated = set(voter['ratings'].keys())
  author_ratings, voter_ratings = vectorize(author['ratings'],
      voter['ratings'])
  features['common_rated'] = len(author_rated.intersection(voter_rated))
  features['jacc_rated'] = jaccard(author_rated, voter_rated)
  features['cos_ratings'] = cosine(author_ratings, voter_ratings)
  pearson = pearsonr(author_ratings, voter_ratings)[0]  
  features['pear_ratings'] = pearson if not isnan(pearson) else 0.0 
  if not author['ratings'] or not voter['ratings']:
    features['diff_avg_ratings'] = nan 
    features['diff_max_ratings'] = nan
    features['diff_min_ratings'] = nan
  else: 
    features['diff_avg_ratings'] = author['avg_rating'] - voter['avg_rating']
    features['diff_max_ratings'] = max(author['ratings'].values()) - \
        max(voter['ratings'].values())
    features['diff_min_ratings'] = min(author['ratings'].values()) - \
        min(voter['ratings'].values()) 
  return features


def calculate_connection_strength(author, voter, trusts, katz, nodes_index):
  """ Calculates connection strength features between author and voter.

      Args:
        author: a dictionary with author containing individual user features.
        voter: a dictionary with voter containing individual user features.
      
      Returns:
        A dictionary with features represeting connection strength in trust
      netwoek between author and voter.
  """
  features = {}
  if not author or not voter or not author['id'] in trusts or voter['id'] not \
      in trusts:
    features['jacc_trustees'] = nan 
    features['jacc_trustors'] = nan
    features['adamic_adar_trustees'] = nan
    features['adamic_adar_trustors'] = nan
    features['katz'] = nan
  else:   
    author_trustees = set(trusts.successors(author['id']))
    voter_trustees = set(trusts.successors(voter['id']))
    author_trustors = set(trusts.predecessors(author['id']))
    voter_trustors = set(trusts.predecessors(voter['id']))
    features['jacc_trustees'] = jaccard(author_trustees, voter_trustees)
    features['jacc_trustors'] = jaccard(author_trustors, voter_trustors)
    features['adamic_adar_trustees'] = adamic_adar_trustees(trusts,
        author['id'], voter['id'])
    features['adamic_adar_trustors'] = adamic_adar_trustors(trusts,
        author['id'], voter['id'])
    features['katz'] = 0
    if author['id'] in nodes_index and author['id'] in nodes_index:
      a_index = nodes_index[author['id']]
      v_index = nodes_index[voter['id']]
      features['katz'] = katz[v_index,a_index]
  return features


def model_author_voter_similarity(train, users, test_pairs):
  """ Models users similarities using votes in the training set. 

      Args:
        train: list of votes in the training set.
        users: dictionary of user models.

      Returns:
        A dictionary of similarity features indexed by a pair of user ids.
  """
  sim_features = {}
  for vote in train:
    author_id = vote['author']
    voter_id = vote['voter']
    if (author_id, voter_id) in sim_features:
      continue
    sim_features[(author_id, voter_id)] = \
        calculate_authoring_similarity(users[author_id], users[voter_id])
  for author_id, voter_id in test_pairs:
    if (author_id, voter_id) in sim_features:
      continue
    if author_id in users and voter_id in users:
      sim_features[(author_id, voter_id)] = \
          calculate_authoring_similarity(users[author_id], users[voter_id])
    else:
      sim_features[(author_id, voter_id)] = \
          calculate_authoring_similarity(None, None)

  return sim_features


def model_author_voter_connection(train, users, trusts, test_pairs):
  """ Models users connection strength using votes in the training set. 

      Args:
        train: list of votes in the training set.
        users: dictionary of user models.
        trusts: networkx Digraph with trust network.

      Returns:
        A dictionary of connection features indexed by a pair of user ids.
  """
  conn_features = {}
  katz, nodes_index = get_katz_matrix(trusts) 
  for vote in train:
    author_id = vote['author']
    voter_id = vote['voter']
    if (author_id, voter_id) in conn_features:
      continue
    conn_features[(author_id, voter_id)] = \
        calculate_connection_strength(users[author_id], users[voter_id], trusts,
        katz, nodes_index)
  for author_id, voter_id in test_pairs:
    if (author_id, voter_id) in conn_features:
      continue
    if author_id in users and voter_id in users:
      conn_features[(author_id, voter_id)] = \
          calculate_connection_strength(users[author_id], users[voter_id], trusts,
          katz, nodes_index)
    else:
      conn_features[(author_id, voter_id)] = \
          calculate_connection_strength(None, None, trusts, katz, nodes_index)
  return conn_features

