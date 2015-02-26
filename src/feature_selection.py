""" Feature selection module
    ------------------------

    Performs feature selection using three methods: Tree-based, RFE and
    InfoGain.

    Usage:
      $ python -m src.feature_selection
    on root directory of the project.
"""


from math import log

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

from src.prediction import read


_TRAIN_FILE = '/var/tmp/luciana/train20.csv'
_TEST_FILE = '/var/tmp/luciana/test20.csv'


""" Ranks the features using a tree-based method.

    Args:
      features: a list with the feature names, in the order they appear in data.
      data: a list with numpy array represeting the instances.
      truth: a list with integers, ranging from 0 to 5, with the helpfulness
    votes for each vote instance.

    Returns:
      A list with feature names sorted by importance.
"""
def rank_features_tree(features, data, truth):
  pred = RandomForestClassifier()
  pred.fit(data, truth)
  importances = pred.feature_importances_ # the higher the better
  ranking = sorted(features, key=lambda x: importances[features.index(x)],
    reverse=True)
  return ranking


""" Ranks the features using RFE (Recursive Feature Elimination) method.

    Args:
      features: a list with the feature names, in the order they appear in data.
      data: a list with numpy array represeting the instances.
      truth: a list with integers, ranging from 0 to 5, with the helpfulness
    votes for each vote instance.

    Returns:
      A list with feature names sorted by importance.
"""
def rank_features_rfe(features, data, truth):
  pred = LinearRegression()
  rfe = RFE(estimator=pred, n_features_to_select=1, step=1)
  rfe.fit(data, truth)
  importances = rfe.ranking_ # the lower the better
  ranking = sorted(features, key=lambda x: importances[features.index(x)])
  return ranking


""" Gets a probability distribution from a collection of values. 

    Args:
      collection: a list representing a collection of values with repetition.

    Returns:
      A dictionary indexed by each value in collection and containing the
    probability frequency divided by size of collection).
"""
def get_probability_distribution(collection):
  freq = {}
  size = len(collection)
  for value in collection:
    if value not in freq:
      freq[value] = 0.0
    freq[value] += 1.0
  return {value: freq[value] / size for value in freq}


""" Calculates the entropy of a collection. The entropy is a metric opposite to
    purity.

    Args:
      collection: a list with values to calculate the entropy of.

    Returns:
      A real value representing the entropy.
"""
def calculate_entropy(collection):
  entropy = 0.0
  prob = get_probability_distribution(collection)
  return sum([- prob[value] * log(prob[value], 2) for value in prob])


""" Ranks the features using Information Gain method. 

    Args:
      features: a list with the feature names, in the order they appear in data.
      data: a list with numpy array represeting the instances.
      truth: a list with integers, ranging from 0 to 5, with the helpfulness
    votes for each vote instance.

    Returns:
      A list with feature names sorted by importance.
"""
def rank_features_infogain(features, data, truth):
  info_gain = []
  size = len(truth)
  old_entropy = calculate_entropy(truth)
  for index in range(len(data[0])):
    value_prob = get_probability_distribution([inst[index] for inst in data])
    partitions = {v: [truth[i] for i in range(size) if data[i][index] == v]
        for v in value_prob}
    new_entropy = 0.0
    for value, prob in value_prob.items():
      new_entropy += prob * calculate_entropy(partitions[value])
    info_gain.append(old_entropy - new_entropy)
  return sorted(features, key=lambda f: info_gain[features.index(f)],
      reverse=True)


""" Evaluate non-personalized and personalized features using two methods: a
    tree-based one, using Random Forest Classifier, and RFE (Recursive Feature
    Elimination) in linear regression.

    Args:
      features: a list with features names, in order as they appear in data.
      data: a list of instances, represented as numpy arrays continaing certain
    features.
      truth: a list of integers, from 0 to 5, containing the correct values for
    the instances.

    Returns:
      None. The results are output to stdout.
"""
def evaluate_features(features, data, truth):
  print 'Tree-based Feature Evaluation'
  tree_ranking = rank_features_tree(features, data, truth)
  for index, feature in enumerate(tree_ranking):
    print '%d. %s' % (index, feature)
  print '-----------------------------'

  print 'RFE Feature Evaluation'
  rfe_ranking = rank_features_rfe(features, data, truth)
  for index, feature in enumerate(rfe_ranking):
    print '%d. %s' % (index, feature)
  print '-----------------------------'

#  print 'InfoGain Feature Evaluation'
#  ig_ranking = rank_features_infogain(features, data, truth)
#  for index, feature in enumerate(ig_ranking):
#    print '%d. %s' % (index, feature)
#  print '-----------------------------'


if __name__ == '__main__':
  features, _, _, train_p, train_truth = read(_TRAIN_FILE)
  _, _, _, test_p, test_truth = read(_TEST_FILE)
  evaluate_features(features, train_p + test_p, train_truth + test_truth)

