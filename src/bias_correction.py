""" Bias correction module 
    ----------------------

    Module for removing bias from intended targets, which can be a combination
    of review, author and voter. 

    Usage:
      Used only as a module, not directly callable.
"""


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, MinMaxScaler


_IDS_STOP = 3 # index + 1 where id features end


""" Parses a target code, inserted as a command line argument, into a list of
    target entities for the bias correction.

    Args:
      target_code: a string containing from one to three characters, 'r' for
    review entity, 'a' for author, and 'v' for voter.

    Returns:
      A list of strings with the target names.
"""
def parse_target(target_code):
  targets = []
  if target_code.contains('r'):
    targets.append('review')
  if target_code.contains('a'):
    targets.append('author')
  if target_code.contains('v'):
    targets.append('voter')
  if not targets:
    print ('Error: bias correction should contain at least one character in the'
           'set {\'r\', \'a\', \'v\'}')
    import sys
    sys.exit()
  return targets


""" Gets entities bias from data. The bias consists on the average vote value
    deviation from the mean related to the entity.

    Args:
      ids: instances' ids for review, author and voter.
      truth: truth value per instance.
      targets: entities whose bias should be considered.

    Returns:
      A dictionary with biases, first indexed by entity type (target) and
    second, by entity id.
"""
def get_entities_biases(ids, truth, targets):
  entities = {target: {} for target in targets}
  bias = {target: {} for target in targets}
  bias['avg'] = float(sum(truth)) / len(truth)
  
  for index, instance in enumerate(ids):
    ids = {}
    ids['review'], ids['author'], ids['voter'] = instance[:_IDS_STOP]
    for etype in targets:
      entity = ids[etype]
      edict = entities[etype]
      if entity not in edict:
        edict[entity] = {}
        edict[entity]['sum'], edict[entity]['count'] = 0,0
      edict[entity]['sum'] += truth[index]
      edict[entity]['count'] += 1
  for etype in targets:
    edict = entities[etype]
    for entity in edict:
      edict[entity] = float(edict[entity]['sum']) / edict[entity]['count']
      edict[entity] -= bias['avg']

  return bias


""" Updates truth values removing biases.

    Args:
      ids: a list of arrays with the ids of review, author and voter of 
    instances.
      truth: a list of integers with truth values associated to instances.
      targets: a list of strings with the entities targets to consider bias.
      bias: a dictionary indexed by target and then by target id containing bias
    values. It also contains a special key 'avg' with the global average vote.

    Returns:
      A list of real values containing new truth values after bias removal.
"""
def update_truth(ids, truth, targets, bias):
  new_truth = [0] * len(truth)
  for index, instance in enumerate(ids):
    ids = {}
    ids['review'], ids['author'], ids['rater'] = instance[:_IDS_STOP]
    new_truth[index] = truth[index] - bias['avg']
    for etype in targets:
      entity = ids[etype]
      new_truth[index] -= bias[etype][entity]
  return new_truth


""" Removes bias from helpfulness votes. The bias may correspond to review's,
    author's or voter's helpfulness average.

    Args:
      ids: a list of arrays containing the ids of the votes instances (review,
        author, voter).
      truth: a list containing the helpfulness votes associated to the
        instances.
      target_code: a string containing from one to three characters, 'r' for
    review entity, 'a' for author, and 'v' for voter.

    Returns:
      A bias dictionary, with global helpfulness average, voters biases, reviews
    biases and authors biases, and a list of truth values with bias removed. 
"""
def remove_bias(ids, truth, target_code):
  targets = parse_target(target_code)
  bias = get_entities_biases(ids, truth, targets) 
  new_truth = update_truth(ids, truth, targets, bias)
  return bias, new_truth


""" Adjusts predicted truth values accounting for bias.

    Args:
      bias: a dictionary indexed by target and then by target id containing bias
    values. It also contains a special key 'avg' with the global average vote.
      ids: list of arrays containing ids (review, author, voter) for the
    instances.
      res: list of predicted results to adjust with bias.
      target_code: a string containing from one to three characters, 'r' for
    review entity, 'a' for author, and 'v' for voter.

    Returns:
      A list of predicted values after bias ajust.
"""
def adjust_bias(bias, ids, res, target_code):
  targets = parse_target(target_code)
  new_res = res[:]
  for index, instance in enumerate(ids):
    ids = {}
    ids['review'], ids['author'], ids['voter'] = instance[:_IDS_STOP]
    new_res[index] += bias['avg']
    for etype in targets:
      entity = ids[etype]
      bdict = bias[etype]
      new_res[index] += bdict[entity]
  return new_res

