""" Predictor Module
    ----------------

    Realizes the task of pred_code: a code for the predictor, used for indexing
    the dictionary _PREDICTORS.

    Usage:
      $ python -m src.prediction -p <predictor> [-b <bias_code>]
        [-s <scaling_type>] [-i <iteration_count>]
    in the root directory of the project. In this command,
      (*) <predictor> is a code for the predictor to use; it is mandatory and
    should be one key defined in _PREDICTORS;
      (*) <bias> is a string with one to three characters containing a
    subset of {'r', 'a', 'v'}, representing, respectively, review, author and
    voter biases to target;
      (*) <scaling_type> can be 'standard' for zero mean and unit variance, or
    'minmax' for a range between 0 and 1;
      (*) <iteration_count> is a positive integer with the intended number of
    predictor fit and evaluation iterations (useful when the predictor involves
    randomness, such as Random Forest Classifier).
"""


from csv import reader
from math import sqrt
from sys import argv

from numpy import array
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

from src.prediction_evaluation import calculate_rmse
from src.bias_correction import remove_bias, adjust_bias
from src.scaling import scale_features 


_TRAIN_FILE = '/var/tmp/luciana/train20.csv'
_TEST_FILE = '/var/tmp/luciana/test20.csv'
_IDS_STOP = 3 # index + 1 where id features end
_NP_STOP = 34 # index + 1 where non-personalized features end
_FEAT_STOP = 46 # index + 1 where all the features end
_PREDICTORS = {'svm': SVC, 'lsvm': LinearSVC, 'svr': SVR,
    'rfr': RandomForestRegressor, 'rfc': RandomForestClassifier,
    'dtc': DecisionTreeClassifier, 'dtr': DecisionTreeRegressor,
    'lr': LinearRegression, 'gnb': GaussianNB}
_Z = 1.959964 # 95% of confidence for IC with both sides


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
      data_np.append(array([float(f) for f in row[_IDS_STOP:_NP_STOP]]))
      if selected_features:
        data_p.append(array([float(row[i]) for i in range(len(row)) if i in
            selected_indices]))
      else:
        data_p.append(array([float(f) for f in row[_IDS_STOP:_FEAT_STOP]]))
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
  if pred_code == 'SVR' or pred_code == 'SVM':
    pred = _PREDICTORS[pred_code](cache_size=2000)
  else:
    pred = _PREDICTORS[pred_code]()
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


""" Prints summary of scores, with confidence interval, for a set of
    predictions' runs.

    Observations:
      It is considered that at least 30 iterations are performed, thus allowing
    no assumption about data distribution.

    Args:
      scores_np: list with iterations scores for non-personalized version.
      scores_p: list with iterations scores for personalized version.

    Returns:
      None. The output is displayed on stdout.
"""
def print_summary(scores_np, scores_p):
    print '*******************'
    print 'SUMMARY'
    mean_np = sum(scores_np) / len(scores_np)
    std_np = std(scores_np, ddof=1)
    width_np = _Z * std_np / sqrt(rep)
    print '(*) Non-personalized:'
    print 'Mean RMSE: %f' % mean_np 
    print 'Empiric SD of RMSE: %f' % std_np 
    print 'IC of 95%%: (%f, %f)' % (mean_np - width_np, mean_np + width_np)
    mean_p = sum(scores_p) / len(scores_p)
    std_p = std(scores_p, ddof=1)
    width_p = _Z * std_p / sqrt(rep)
    print ''
    print '(*) Personalized:'
    print 'Mean RMSE: %f' % mean_p 
    print 'Empiric SD of RMSE: %f' % std_p
    print 'IC of 95%%: (%f, %f)' % (mean_p - width_p, mean_p + width_p)
    print '*******************'


""" Main function of prediction module. Performs feature evaluation and
    perfomance comparisson between non-personalized and personalized predictors.

    Args:
      pred_code: the code for the chosen predictor, using to index _PREDICTORS.
      scale: scale type to be performed or None.
      bias_code: bias code or None.
      rep: number of repetitions of prediction to run.

    Returns:
      None. The results are output to stdout.
"""
def main(pred_code, scale, bias_code, rep):
  scores_np, scores_p = [], []
  
  features, train_ids, train_np, train_p, train_truth = read(_TRAIN_FILE,
      _SEL_FEAT)
  _, test_ids, test_np, test_p, test_truth = read(_TEST_FILE, _SEL_FEAT)

  if scale:
    train_np, test_np = scale_features(train_np, test_np)
    train_p, test_p = scale_features(train_p, test_p)
  
  for _ in range(rep):
    if bias_code:
      bias, bias_train_truth = remove_bias(train_ids, train_truth, bias_code)
      pred_np = train(train_np, bias_train_truth, pred_code)
      pred_p = train(train_p, bias_train_truth, pred_code)
    else:
      pred_np = train(train_np, train_truth, pred_code)
      pred_p = train(train_p, train_truth, pred_code)

    res_np, res_p = test(test_np, pred_np), test(test_p, pred_p)

    if bias_code:
      res_np = adjust_bias(bias, test_ids, res_np, bias_code)
      res_p = adjust_bias(bias, test_ids, res_p, bias_code)

    rmse_np, rmse_p = compare(res_np, res_p, test_truth)
    scores_np.append(rmse_np)
    scores_p.append(rmse_p)

  if rep > 1:
    print_summary(scores_np, scores_p)


if __name__ == '__main__':
  pred = None
  scale = None
  bias_code = None
  rep = 1
  for i in range(1, len(argv), 2):
    if argv[i] == 'p':
      pred = argv[i+1]
    elif argv[i] == 's':
      scale = argv[i+1]
    elif argv[i] == 'b':
      bias_code = argv[i+1]
    elif argv[i] == 'i':
      rep = int(argv[i+1])
  main(pred, scale, bias_code, rep)
