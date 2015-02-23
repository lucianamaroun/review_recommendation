""" Repeated feature deletion module
    --------------------------------

    Evaluates a predictor repeatedly by removing one least import feature at a
    time.

    Usage:
      $ python -m src.rep_feature_removal -p <predictor> [-b <bias_code>]
        [-s <scaling_type>] -r <ranking_method>
    on root directory of the project. In this command,
      (*) <predictor> is a code for the predictor; one in the set of keys of
    dictionary _PREDICTORS;
      (*) <bias> is a string with one to three characters containing a
    subset of {'r', 'a', 'v'}, representing, respectively, review, author and
    voter biases to target;
      (*) <scaling_type> can be 'standard' for zero mean and unit variance, or
    'minmax' for a range between 0 and 1;
      (*) <ranking_method> is a code for the method to rank features. The
    possibilities are 'tree', for Tree-based; 'rfe', for Repeated Feature
    Elimination; or 'infogain', for Information Gain.
"""


from sys import argv

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

from src.feature_selection import rank_features_tree, rank_features_rfe, \
    rank_features_infogain


_PREDICTORS = {'svm': SVC, 'lsvm': LinearSVC, 'svr': SVR,
    'rfr': RandomForestRegressor, 'rfc': RandomForestClassifier,
    'dtc': DecisionTreeClassifier, 'dtr': DecisionTreeRegressor,
    'lr': LinearRegression, 'gnb': GaussianNB}
_RANK_METHODS = {'tree': rank_features_tree, 'rfe': rank_features_rfe,
    'infogain': rank_features_infogain}


""" Repeatedly remove features, from less import to most important, until only
    one is left.

    Args:
      pred_code: code for predictor.
      scale: scale type, if any, otherwirse None.
      bias_code, bias code, if any, otherwise None.
      rank_type: method of feature ranking to performing, either 'tree', 'rfe'
    or 'infogain'.

    Returns:
      None. The score for each feature removal step is output to stdout.
"""
def repeatedly_remove(pred_code, scale, bias_code, rank_type):
  features, train_ids, _, train, train_truth = read(_TRAIN_FILE)
  _, test_ids, _, test, test_truth = read(_TEST_FILE)
  
  if scale:
    train, test = scale_features(train, test)
  
  cur_features = features[:] + 'dummy' # dummy to remove none at first
  cur_train, cur_test = train, test
  feat_ranking = _RANK_METHODS[rank_type](features, train + test, train_truth +
      test_truth)
  feat_ranking.append('dummy')

  for removal_feature in reversed(feat_ranking[1:]):
    removal_index = cur_features.index(removal_feature)
    cur_features.pop(removal_index)
    cur_train = [array(instance.tolist()[:removal_index] +
        instance.tolist()[removal_index+1:]) for instance in cur_train]
    cur_test = [array(instance.tolist()[:removal_index] +
        instance.tolist()[removal_index+1:]) for instance in cur_test]
    if bias_code:
      bias, bias_train_truth = remove_bias(train_ids, train_truth, bias_code)
      pred = train(train, bias_train_truth, pred_code)
    else:
      pred = train(train, train_truth, pred_code)
    res = test(test, pred)
    if bias_code:
      res = adjust_bias(bias, test_ids, res, bias_code)

    rmse_p = calculate_rmse(res_p, test_truth)
    print 'TOP %d: %f' % (len(cur_features), rmse_p)


if __name__ == '__main__':
  pred = None
  scale = None
  bias_code = None
  rank_type = None
  for i in range(1, len(argv), 2):
    if argv[i] == 'p':
      pred = argv[i+1]
    elif argv[i] == 's':
      scale = argv[i+1]
    elif argv[i] == 'b':
      bias_code = argv[i+1]
    elif argv[i] == 'r':
      rep = int(argv[i+1])
  repeatedly_remove(pred, scale, bias_code, rank_type)
