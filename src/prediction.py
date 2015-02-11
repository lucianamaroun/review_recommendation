""" Predictor Module
    ----------------

    Realizes the task of pred_code: a code for the predictor, used for indexing
    the dictionary _PREDICTORS.

    Returns:
      A scikit learn classifier object.
"""
from csv import reader
from math import sqrt
from sys import argv

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE


_TRAIN_FILE = '/var/tmp/luciana/train10.csv'
_TEST_FILE = '/var/tmp/luciana/test10.csv'
_IDS_STOP = 3 # index + 1 where id features end
_NP_STOP = 36 # index + 1 where non-personalized features end
_FEAT_STOP = 48 # index + 1 where all the features end
_PREDICTORS = {'svm': SVC, 'rfc': RandomForestClassifier,
    'lr': LinearRegression, 'lrb': LinearRegression, 'svr': SVR, 'svrb': SVR,
    'lsvm': LinearSVC,
    'rfr': RandomForestRegressor, 'rfrb': RandomForestRegressor,
    'dtc': DecisionTreeClassifier, 'dtr': DecisionTreeRegressor,
    'dtrb': DecisionTreeRegressor, 'gnb': GaussianNB}
_SEL_FEAT = None
   # set(['r_avg_help_rec', 'num_words', 'u_avg_rel_help_giv',
   # 'u_avg_help_giv', 'num_tokens', 'u_sd_help_giv', 'r_sd_help_rec',
   # 'num_chars', 'u_avg_help_rec', 'u_pagerank', 'u_num_trustors', 
   # 'u_sd_help_rec', 'u_num_trustees', 'num_sents', 'r_num_trustees',
   # 'unique_ratio', 'u_avg_rating', 'kl_div', 'avg_sent', 'adj_ratio',
   # 'noun_ratio', 'r_num_trustors', 'num_ratio', 'neg_sent', 'adv_ratio',
   # 'u_num_reviews', 'cap_ratio', 'verb_ratio', 'pos_sent', 'sym_ratio',
   # 'u_sd_rating', 'r_avg_rel_help_giv', 'r_avg_help_giv', 'trust',
   # 'r_sd_help_giv', 'r_pagerank', 'r_avg_rating', 'r_num_reviews', 'rating',
   # 'r_sd_rating', #'comp_ratio', 'fw_ratio', 'punct_ratio', 'pos_ratio',
   # 'neg_ratio'
   # ])

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
    csv_reader = reader(data)
    features = csv_reader.next()[_IDS_STOP:_FEAT_STOP] # header
    if selected_features:
      selected_indices = set([i+3 for i in range(len(features)) if features[i] in
          selected_features])
    for row in csv_reader:
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
  pred = _PREDICTORS[pred_code](cache_size=1000)
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
  rmse = sqrt(float(sum(errors)) / len(errors))
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


""" Removes bias from helpfulness votes. The bias may correspond to review's,
    author's or voter's helpfulness average.

    Args:
      ids: a list of arrays containing the ids of the votes instances (review,
        author, voter).
      truth: a list containing the helpfulness votes associated to the
        instances.

    Returns:
      Global helpfulness average, dictionary of voter bias, dictionary of
    review bias, dictionary of author bias and a list of truth values with bias
    removed. 
"""
def remove_bias(ids, truth):
  raters = {}
  reviews = {}
  authors = {}
  rtr_bias = {}
  rev_bias = {}
  aut_bias = {}
  all_sum = 0
  for index, instance in enumerate(ids):
    rater = instance[2] # rater id
    author = instance[1] # author id
    review = instance[0] # review id
    if rater not in raters:
      raters[rater] = {}
      raters[rater]['sum'] = 0
      raters[rater]['count'] = 0
    raters[rater]['sum'] += truth[index] # truth
    raters[rater]['count'] += 1
    if author not in authors:
      authors[author] = {}
      authors[author]['sum'] = 0
      authors[author]['count'] = 0
    authors[author]['sum'] += truth[index] # truth
    authors[author]['count'] += 1
    if review not in reviews:
      reviews[review] = {}
      reviews[review]['sum'] = 0
      reviews[review]['count'] = 0
    reviews[review]['sum'] += truth[index] # truth
    reviews[review]['count'] += 1
  all_avg = float(sum(truth)) / len(truth)
  for rtr in raters:
    rtr_bias[rtr] = float(raters[rtr]['sum']) / raters[rtr]['count'] - all_avg
  for aut in authors:
    aut_bias[aut] = float(authors[aut]['sum']) / authors[aut]['count'] - all_avg
  for rev in reviews:
    rev_bias[rev] = float(reviews[rev]['sum']) / reviews[rev]['count'] - all_avg
  new_truth = [0] * len(truth)
  for index, instance in enumerate(ids):
    rtr = instance[2] # rater id
    aut = instance[1] # author id
    rev = instance[0] # review id
    new_truth[index] = truth[index] - all_avg - rev_bias[rev] - rtr_bias[rtr]
       
  return all_avg, rtr_bias, rev_bias, aut_bias, new_truth


""" Adjusts predicted truth values accounting for bias.

    Args:
      avg: global average of helpfulness.
      rtr_bias: dictionary with voter bias indexed by voter id.
      rev_bias: dictionary with review bias indexed by review id.
      aut_bias: dictionary with author bias indexed by author id.
      ids: list of arrays containing ids (review, author, voter) for the
        instances.
      res: list of predicted results to adjust with bias.

    Returns:
      A list of predicted values after bias ajust.
"""
def adjust_bias(avg, rtr_bias, rev_bias, aut_bias, ids, res):
  new_res = res[:]
  for index, instance in enumerate(ids):
    rater = instance[2]
    author = instance[1]
    review = instance[0]
    new_res[index] += avg
    if rater in rtr_bias: # not cold-start rater
      new_res[index] += rtr_bias[rater]
#    if author in aut_bias: # not cold-start author
#      new_res[index] += aut_bias[author]
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
  rmse_np = calculate_rmse(res_np, test_truth)
  print 'RMSE: %f' % rmse_np 
  print '-----------------------------'

  print 'Personalized Performance:'
  rmse_p = calculate_rmse(res_p, test_truth)
  print 'RMSE: %f' % rmse_p 
  print '-----------------------------'

  return rmse_np, rmse_p


""" Filters features according to a selected set.

    Args:
      features: a list of strings containing features names.
      data: list of instances modeled as vectors.
      selected: a set of strings with the selected features names.
    
    Returns:
      A list of instance modeled as vectors projected on the selected features.
"""
def filter_features(features, data, selected):
  new_data = []
  sel_columns = set()
  for index, feature in enumerate(features):
    if feature in selected:
      sel_columns.add(index)
  for instance in data:
    instance = instance.tolist()
    instance = [instance[i] for i in range(len(instance)) if i in sel_columns]
    instance = array(instance)
    new_data.append(instance)
  return new_data


""" Evaluates linear fit of the data.

    Args:
      data_np: a list of instances containing only non-personalized features.
      data_p: a list of instances containing both non-personalized and
        personalized features.
      truth: a list of truth values associated to instances.

    Returns:
      None. The result is output to stdout and contains R-squared evaluation
    over non-personalized and personalized models.
"""
def evaluate_linear_fit(data_np, data_p, truth):
  lr_np = LinearRegression()
  lr_np.fit(data_np, truth)
  print 'Linear Fit Evaluation'
  print 'Non-personalized'
  print 'R-squared: %f' % lr_np.score(data_np, truth)
  lr_p = LinearRegression()
  lr_p.fit(data_p, truth)
  print 'Personalized'
  print 'R-squared: %f' % lr_p.score(data_p, truth)
  print '-----------------------------'


""" Main function of prediction module. Performs feature evaluation and
    perfomance comparisson between non-personalized and personalized predictors.

    Args:
      pred_code: the code for the chosen predictor, using to index _PREDICTORS.

    Returns:
      None. The results are output to stdout.
"""
def main(pred_code, rep=1):
  scores_p = []
  scores_np = []
  for _ in range(rep):
    features, train_ids, train_np, train_p, train_truth = read(_TRAIN_FILE,
        _SEL_FEAT)
    _, test_ids, test_np, test_p, test_truth = read(_TEST_FILE, _SEL_FEAT)

    if pred_code == 'eval':
      evaluate_features(features, train_p + test_p, train_truth + test_truth)
      evaluate_linear_fit(train_np + test_np, train_p + test_p, train_truth +
          test_truth)
      return

    if pred_code[-1] == 'b':
      avg, rtr_bias, rev_bias, aut_bias, train_truth = remove_bias(train_ids,
          train_truth)

    pred_np = train(train_np, train_truth, pred_code)
    pred_p = train(train_p, train_truth, pred_code)

    res_np = test(test_np, pred_np)
    res_p = test(test_p, pred_p)

    if pred_code[-1] == 'b':
      res_np = adjust_bias(avg, rtr_bias, rev_bias, aut_bias, test_ids, res_np)
      res_p = adjust_bias(avg, rtr_bias, rev_bias, aut_bias, test_ids, res_p)

    rmse_np, rmse_p = compare(res_np, res_p, test_truth)
    scores_np.append(rmse_np)
    scores_p.append(rmse_p)

  Z = 1.959964
  if rep > 1:
    print '*******************'
    print 'SUMMARY'
    mean_np = sum(scores_np) / len(scores_np)
    std_np = std(scores_np, ddof=1)
    width_np = Z * std_np / sqrt(rep)
    print 'Non-personalized:'
    print 'Measures: %s' % scores_np
    print 'Mean RMSE: %f' % mean_np 
    print 'Empiric SD of RMSE: %f' % std_np 
    print 'IC of 95%%: (%f, %f)' % (mean_np - width_np, mean_np + width_np)
    mean_p = sum(scores_p) / len(scores_p)
    std_p = std(scores_p, ddof=1)
    width_p = Z * std_p / sqrt(rep)
    print 'Personalized:'
    print 'Measures: %s' % scores_p
    print 'Mean RMSE: %f' % mean_p 
    print 'Empiric SD of RMSE: %f' % std_p
    print 'IC of 95%%: (%f, %f)' % (mean_p - width_p, mean_p + width_p)
    print '*******************'


if __name__ == '__main__':
  main(argv[1] if len(argv) > 1 else 'rfc', int(argv[2]) if len(argv) > 2 else 1)
