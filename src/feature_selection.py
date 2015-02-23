""" Feature selection module
    ------------------------

    Performs feature selection using three methods: Tree-based, RFE and
    InfoGain.

    Usage:
      $ python -m src.feature_selection
    on root directory of the project.
"""


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


_TRAIN_FILE = '/var/tmp/luciana/train20.csv'
_TEST_FILE = '/var/tmp/luciana/test20.csv'


""" Ranks the features using a tree-based method.

    Args:
      features: the list with the feature names, in the order they appear in
    data.
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
      features: the list with the feature names, in the order they appear in
    data.
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


""" Ranks the features using Information Gain method. 

    Args:
      features: the list with the feature names, in the order they appear in
    data.
      data: a list with numpy array represeting the instances.
      truth: a list with integers, ranging from 0 to 5, with the helpfulness
    votes for each vote instance.

    Returns:
      A list with feature names sorted by importance.
"""
def rank_features_infogain(features, data, truth):
  # TODO
  pass


""" Evaluate non-personalized and personalized features using two methods: a
    tree-based one, using Random Forest Classifier, and RFE (Recursive Feature
    Elimination) in linear regression.

    Args:
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

  return tree_ranking, rfe_ranking


if __name__ == '__main__':
  features, _, _, train_p, train_truth = read(_TRAIN_FILE)
  _, _, _, test_p, test_truth = read(_TEST_FILE)
  evaluate_features(features, train_p + test_p, train_truth + test_truth)

