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
from sys import argc, argv

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.feature_selection import RFE


_TRAIN_FILE = '/var/tmp/luciana/train-notustat.txt'
_TEST_FILE = '/var/tmp/luciana/test-notustat.txt'
_IDS_STOP = 3 # index + 1 where id features end
_NP_STOP = 25 # index + 1 where non-personalized features end
_FEAT_STOP = 32 # index + 1 where all the features end
_PREDICTORS = {'svm': SVC, 'rfc': RandomForestClassifier,
    'lg': LinearRegression}


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
def read(data_file):
  data_np = []
  data_p = []
  truth = []
  with open(data_file, 'r') as data:
    reader = csv.reader(data)
    features = reader.next()[_IDS_STOP:_FEAT_STOP] # header
    for row in raters_reader:
      data_np.append(np.array([float(f) for f in row[_IDS_STOP:_NP_STOP]]))
      data_p.append(np.array([float(f) for f in row[_IDS_STOP:_FEAT_STOP]]))
      truth.append(int(row[_FEAT_STOP]))
  return features, data_np, data_p, truth


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
  pred = _PREDICTORS[pred_code]()
  pred.fit(train, truth)
  return pred


""" Tests a predictor.

    Args:
      test: a list of numpy arrays with the votes instances used as test.
      clf: a scikit learn classifier object.

    Returns:
      An array with predicted values, integers ranging from 0 to 5, for the
    test instances.
"""
def test(test, clf):
  prediction = clf.predict(test)
  return prediction


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
def evaluate_features(data, truth):
  print 'Tree-based Feature Evaluation'
  for index, feature in enumerate(rank_features_tree(data, truth)):
    print '%d. %s' % (index, feature)
  print '-----------------------------'

  print 'RFE Feature Evaluation'
  for index, feature in enumerate(rank_features_rfe(data, truth)):
    print '%d. %s' % (index, feature)
  print '-----------------------------'


""" Calculates squared errors.

    Args:
      prediction: a list of predicted values assigned to the instances.
      truth: a list with the corrected values.

    Returns:
      A list of squared errros, one for each instance.
"""
def calculate_squared_errors(prediction, truth):
  errors = [(prediction[i] - truth[i]) ** 2 for i in range(len(prediction))]
  return errors


""" Calculates RMSE (Root Mean Squared Error) of a prediction task.

    Args:
      prediction: a list of predicted values assigned to the instances.
      truth: a list with the corrected values.

    Returns:
      A real value representing the RMSE.
"""
def calculate_rmse(prediction, truth):
  errors = calculate_squared_errors(prediction, truth)
  rmse = math.sqrt(float(sum(errors)) / len(errors))
  return rmse


""" Calculates the accuracy of a prediction task.

    Args:
      prediction: a list with the predicted values for the instances.
      truth: a list with the correct values.

    Returns:
      A real value representing the accuracy of the method.
"""
def calculate_accuracy(prediction, truth):
  errors = calculate_squared_errors(prediction, truth)
  binary_errors = [1 if e > 0 else 0 for e in errors]
  accuracy = float(sum(binary_errors)) / len(binary_errors)
  return accuracy


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
  clf = ExtraTreesClassifier()
  clf.fit(data, truth)
  importances = clf.feature_importances_ # the higher the better
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
  clf = linear_model.LinearRegression()
  rfe = RFE(estimator=clf, n_features_to_select=1, step=1)
  rfe.fit(data, truth)
  importances = rfe.ranking_ # the lower the better
  ranking sorted(features, key=lambda x: importances[features.index(x)])
  return ranking


""" Compares non-personalized with personalized prediction of helpfulness votes.

    Args:
      train_np: a list of numpy arrays representing train instances with
    non-personalized features.
      train_p: a list of numpy arrays representing train instances with
    personalized features.
      train_truth: a list with the correct helpfulness values of the train
    instances.
      test_np: a list of numpy arrays representing test instances with
    non-personalized features.
      test_p: a list of numpy arrays representing test instances with
    personalized features.
      test_truth: a list with the correct helpfulness values of the test
    instances.
      pred_code: the code indicating the predictor choice used to index
    _PREDICTORS dictionary.

    Returns:
      None. The results are output to stdout.
"""
def compare(train_np, train_p, train_truth, test_np, test_p, test_truth,
    pred_code):
  clf_np = train(train_np, train_truth, pred_code)
  clf_p = train(train_p, train_truth, pred_code)

  pred_np = test(test_np, clf_np)
  pred_p = test(test_p, clf_p)

  print 'Non-Personalized Performance:'
  print 'RMSE: %f' % calculate_rmse(pred_np, test_truth)
  print 'Accuracy: %f' % calculate_accuracy(pred_np, test_truth)
  print '-----------------------------'

  print 'Personalized Performance:'
  print 'RMSE: %f' % calculate_rmse(pred_np, test_truth)
  print 'Accuracy: %f' % calculate_accuracy(pred_np, test_truth)
  print '-----------------------------'


""" Main function of prediction module. Performs feature evaluation and
    perfomance comparisson between non-personalized and personalized predictors.

    Args:
      pred_code: the code for the chosen predictor, using to index _PREDICTORS.

    Returns:
      None. The results are output to stdout.
"""
def main(pred_code):
  features, train_np, train_p, train_truth = read(_TRAIN_FILE)
  _, test_np, test_p, test_truth = read(_TEST_FILE)

  evaluate_features(features, train_p + test_p, train_truth + test_truth)

  compare(train_np, train_p, train_truth, test_np, test_p, test_truth, pred_code)


if __name__ == '__main__':
  main(argv[1] if argc > 1 else 'rfc') # use Random Forest as default
