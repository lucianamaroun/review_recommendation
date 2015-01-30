""" Predictor Module
    ----------------

    Realizes the task of predicting heplfulness votes and evaluating the
  results.

    Usage:
      $ python -m src.prediction <predictor_code>
    on the project root directory.
    <predictor_code> can be svm for Support Vector Machine; rfc for Random
  Forest Classifier; or lg for Linear Regression.
"""

import numpy as np
import csv
import math
from sys import argv

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import RFE


_TRAIN_FILE = '/var/tmp/luciana/train-notustat.txt'
_TEST_FILE = '/var/tmp/luciana/test-notustat.txt'
_IDS_STOP = 3 # index + 1 where id features end
_NP_STOP = 25 # index + 1 where non-personalized features end
_FEAT_STOP = 32 # index + 1 where all the features end
_PREDICTORS = {'svm': SVC, 'rfc': RandomForestClassifier,
    'lr': LinearRegression, 'lrb': LinearRegression, 'svr': SVR,
    'rfr': RandomForestRegressor}
_SEL_FEAT = set(['r_avg_help_rec', 'u_avg_rel_help_giv', 'u_avg_help_giv',
    'num_tokens', 'unique_ratio', 'num_sents', 'u_num_trustees',
    'u_num_trustors', 'u_avg_rel_rating', 'u_avg_rating', 'r_num_trustees',
    'noun_ratio', 'avg_sent', 'adj_ratio', 'num_ratio', 'adv_ratio', 
    'verb_ratio', 'cap_ratio', 'r_num_trustors', 'pos_sent', 'r_avg_rel_rating',
    'r_num_reviews', 'neg_sent', 'r_avg_rating', 'trust', 'fw_ratio',
    'comp_ratio', 'sym_ratio', 'punct_ratio'
    ])

""" Reads features and truth values from votes in a data file.

    Args:
      data_file: the string with the csv file with the data.

    Returns:
      Four values: a list with features' names; a list of numpy arrays with
    instances represented by non-personalized features; a list of numpy arrays
    with instances represented by non-personalized and personalized features;
    a list with the truth class for the instances, integers from 0 to 5
    corresponding to helpfulness votes.
"""
def read(data_file, selected_features=None):
  data_np = []
  data_p = []
  ids = []
  truth = []
  with open(data_file, 'r') as data:
    reader = csv.reader(data)
    features = reader.next()[_IDS_STOP:_FEAT_STOP] # header
    if selected_features:
      selected_indices = set([i+3 for i in range(len(features)) if features[i] in
          selected_features])
    for row in reader:
      ids.append(row[:_IDS_STOP])
      data_np.append(np.array([float(f) for f in row[_IDS_STOP:_NP_STOP]]))
      if selected_features:
        data_p.append(np.array([float(row[i]) for i in range(len(row)) if i in
            selected_indices]))
      else:
        data_p.append(np.array([float(f) for f in row[_IDS_STOP:_FEAT_STOP]]))
      truth.append(int(row[_FEAT_STOP]))
  return features, ids, data_np, data_p, truth


""" Trains a predictor.

    Args:
      train: a list of numpy arrays with the votes instances used as train.
      truth: a list of integers containing the truth values of votes, integers
    between 0 and 5.
      pred_code: a code for the predictor, used for indexing the dictionary
    _PREDICTORS.

    Returns:
      A scikit learn classifier object.
"""
def train(train, truth, pred_code):
  pred = _PREDICTORS[pred_code]()#kernel='linear', C=1e3)
  pred.fit(train, truth)
  return pred


""" Tests a predictor.

    Args:
      test: a list of numpy arrays with the votes instances used as test.
      pred: a scikit learn predictor object.

    Returns:
      An array with predicted values, integers ranging from 0 to 5, for the
    test instances.
"""
def test(test, pred):
  result = pred.predict(test)
  return result


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
  for index, feature in enumerate(rank_features_tree(features, data, truth)):
    print '%d. %s' % (index, feature)
  print '-----------------------------'

  print 'RFE Feature Evaluation'
  for index, feature in enumerate(rank_features_rfe(features, data, truth)):
    print '%d. %s' % (index, feature)
  print '-----------------------------'


""" Calculates squared errors.

    Args:
      result: a list of predicted values assigned to the instances.
      truth: a list with the corrected values.

    Returns:
      A list of squared errros, one for each instance.
"""
def calculate_squared_errors(result, truth):
  errors = [(result[i] - truth[i]) ** 2 for i in range(len(result))]
  return errors


""" Calculates RMSE (Root Mean Squared Error) of a prediction task.

    Args:
      result: a list of predicted values assigned to the instances.
      truth: a list with the corrected values.

    Returns:
      A real value representing the RMSE.
"""
def calculate_rmse(result, truth):
  errors = calculate_squared_errors(result, truth)
  rmse = math.sqrt(float(sum(errors)) / len(errors))
  return rmse


""" Calculates the accuracy of a prediction task.

    Args:
      result: a list with the predicted values for the instances.
      truth: a list with the correct values.

    Returns:
      A real value representing the accuracy of the method.
"""
def calculate_accuracy(result, truth):
  errors = calculate_squared_errors(result, truth)
  binary_errors = [1 if e > 0 else 0 for e in errors]
  accuracy = float(sum(binary_errors)) / len(binary_errors)
  return 1 - accuracy


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


def remove_bias(ids, truth):
  raters = {}
  reviews = {}
  rtr_bias = {}
  rev_bias = {}
  all_sum = 0
  for index, instance in enumerate(ids):
    rater = instance[2] # rater id
    review = instance[0] # review id
    if rater not in raters:
      raters[rater] = {}
      raters[rater]['sum'] = 0
      raters[rater]['count'] = 0
    raters[rater]['sum'] += truth[index] # truth
    raters[rater]['count'] += 1
    if review not in reviews:
      reviews[review] = {}
      reviews[review]['sum'] = 0
      reviews[review]['count'] = 0
    reviews[review]['sum'] += truth[index] # truth
    reviews[review]['count'] += 1
    all_sum += truth[index]
  all_avg = float(all_sum) / len(truth)
  for rtr in raters:
    rtr_bias[rtr] = float(raters[rtr]['sum']) / raters[rtr]['count'] - all_avg
  for rev in reviews:
    rev_bias[rev] = float(reviews[rev]['sum']) / reviews[rev]['count'] - all_avg
  new_truth = truth[:]
  for index, instance in enumerate(ids):
    rater = instance[2] # rater id
    review = instance[0] # review id
    new_truth[index] -= all_avg + rtr_bias[rater] + rev_bias[review]
  return all_avg, rtr_bias, rev_bias, new_truth


def adjust_bias(avg, rtr_bias, rev_bias, ids, res):
  new_res = res[:]
  for index, instance in enumerate(ids):
    rater = instance[2]
    review = instance[0]
    new_res[index] += avg
    if rater in rtr_bias: # not cold-start rater
      new_res[index] += rtr_bias[rater]
    if review in rev_bias: # not cold-start review
      new_res[index] += rev_bias[review]
  return new_res


""" Compares non-personalized with personalized prediction of helpfulness votes.

    Args:
      res_np: the predicted values for the non-personalized predictor.
      res_p: the predicted values for the personalized predictor.
      test_truth: the truth values.

    Returns:
      None. The results are output to stdout.
"""
def compare(res_np, res_p, test_truth):

  print 'Non-Personalized Performance:'
  print 'RMSE: %f' % calculate_rmse(res_np, test_truth)
  print '-----------------------------'

  print 'Personalized Performance:'
  print 'RMSE: %f' % calculate_rmse(res_p, test_truth)
  print '-----------------------------'


def filter_features(features, data, selected):
  new_data = []
  sel_columns = set()
  for index, feature in enumerate(features):
    if feature in selected:
      sel_columns.add(index)
  for instance in data:
    instance = instance.tolist()
    instance = [instance[i] for i in range(len(instance)) if i in sel_columns]
    instance = np.array(instance)
    new_data.append(instance)
  return new_data

""" Main function of prediction module. Performs feature evaluation and
    perfomance comparisson between non-personalized and personalized predictors.

    Args:
      pred_code: the code for the chosen predictor, using to index _PREDICTORS.

    Returns:
      None. The results are output to stdout.
"""
def main(pred_code):
  features, train_ids, train_np, train_p, train_truth = read(_TRAIN_FILE)
  _, test_ids, test_np, test_p, test_truth = read(_TEST_FILE)

  #evaluate_features(features, train_p + test_p, train_truth + test_truth)

  if pred_code == 'lrb' or pred_code == 'rfr':
    avg, rtr_bias, rev_bias, train_truth = remove_bias(train_ids, train_truth)
  
  pred_np = train(train_np, train_truth, pred_code)
  pred_p = train(train_p, train_truth, pred_code)

  res_np = test(test_np, pred_np)
  res_p = test(test_p, pred_p)

  if pred_code == 'lrb' or pred_code == 'rfr':
    res_np = adjust_bias(avg, rtr_bias, rev_bias, test_ids, res_np)
    res_p = adjust_bias(avg, rtr_bias, rev_bias, test_ids, res_p)
  
  compare(res_np, res_p, test_truth)


if __name__ == '__main__':
  main(argv[1] if len(argv) > 1 else 'rfc') # use Random Forest as default
