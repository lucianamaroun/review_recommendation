""" Processing Output Module
     i
    ------------------------

    Process output of a prediction and evaluates using RMSE and nDCG.

    Usage:
      python -m evaluation.process_output <set_type> <rep> <predictor> 
    where
    <set_type> indicates whether it is validation or test, coded as 'val' and
      'test', respectively,
    <predictor> is a string with the name of the predictor and parameters
      used in prediction file, 
    <rep> is an indicator value of the presence of repeated executions for each
      configuration of this predictor.
"""


from math import sqrt
from pickle import load
from os.path import isfile
from sys import argv, exit

from numpy import mean, std, array
from scipy.stats import t 
from sklearn.metrics import average_precision_score

from algo.const import NUM_SETS, CONF_QT, RANK_SIZE, REP
from perf.metrics import calculate_rmse, calculate_ndcg, calculate_ap, \
    calculate_err


def parse_args():
  """ Parses and validates command line arguments.

      Args:
        None.

      Returns:
        A string with predictor description, a string with set type and a
      boolean indicating presence of repeated executions.
  """
  if len(argv) != 4:
    print ('Usage: $ python -m evaluation.process_output <set_type> <rep>'
        ' <predictor>')
    exit()
  set_type = argv[1]
  if set_type != 'val' and set_type != 'test':
    print '<set_type> has to be either \'val\' or \'test\''
    exit()
  rep = True if argv[2] == 'y' else False
  if argv[2] not in ['y', 'n']:
    print '<rep> has to be either y or n'
    exit()
  predictor = argv[3]
  return predictor, set_type, rep


def load_data(predictor, set_type, index, rep):
  """ Loads data of predicted and true values to evaluate. 

      Args:
        predictor: a string with the name and configuration of predictor.
        set_type: string with the set type to be used, 'val' for validation set
      and 'test' for test set.
        index: integer with current index of set from list of set splits.
        rep: repetition number of predictor execution.

      Returns:
        A float with sample size and a string with predictor name. 
  """
  if set_type == 'val': 
    votes = load(open('out/pkl/validation-%d.pkl' % index, 'r'))
    predfile = open('out/val/%s-%d-%d.dat'% (predictor, index, rep), 'r')
  else:
    votes = load(open('out/pkl/test-%d.pkl' % index, 'r'))
    predfile = open('out/test/%s-%d-%d.dat'% (predictor, index, rep), 'r')
  reviews = load(open('out/pkl/reviews-%d.pkl' % index, 'r'))
  pred = [float(line.strip()) for line in predfile]
  predfile.close()
  return votes, pred, reviews


def evaluate_regression(pred, votes, output):
  """ Evaluates predicted values using RMSE, a regression metric.

      Args:
        pred: a list of floats with predicted values.
        votes: a list of votes, represented as dictionaries, belonging to
      votes set.
        output: a file object to pirint output on.

      Returns:
        None. The result is printed on output file and stdout.
  """
  truth = [v['vote'] for v in votes]
  rmse = calculate_rmse(pred, truth) 
  print >> output, "RMSE: %f" % rmse
  return rmse

def evaluate_ranking(pred, votes, reviews, output):
  """ Evaluates predicted values using nDCG@K, with K ranging from 1 to 20,
      a ranking metric.

      Args:
        pred: a list of floats with predicted values.
        votes: a list of votes, represented as dictionaries, belonging to
      votes set.
        reviews: a dictionary of reviews.
        output: a file object to print output on.

      Returns:
        None. The result is printed on output file and stdout.
  """
  pred_group = {}
  truth_group = {}
  key_to_reviews = {}
  for i in xrange(len(votes)):
    voter = votes[i]['voter']
    product = reviews[votes[i]['review']]['product']
    key = (voter, product)
    if key not in pred_group:
      key_to_reviews[key] = []
      pred_group[key] = []
      truth_group[key] =[]
    pred_group[key].append(pred[i])
    truth_group[key].append(votes[i]['vote'])
    key_to_reviews[key].append(votes[i]['review'])
  scores = [] 
  goodranks={}
  for i in xrange(1, RANK_SIZE+1):
    sum_scores = 0
    max_score = 0
    max_rank = None
    min_score = float('inf')
    min_rank = None
    for key in pred_group:
      score = calculate_ndcg(pred_group[key], truth_group[key], i)
      sum_scores += score
    score = sum_scores / len(pred_group)
    print >> output, 'NDCG@%d: %f' % (i, score)
    scores.append(score)
  return scores 


def evaluate_ranking_err(pred, votes, reviews, output):
  """ Evaluates predicted values using nDCG@K, with K ranging from 1 to 20,
      a ranking metric.

      Args:
        pred: a list of floats with predicted values.
        votes: a list of votes, represented as dictionaries, belonging to
      votes set.
        reviews: a dictionary of reviews.
        output: a file object to print output on.

      Returns:
        None. The result is printed on output file and stdout.
  """
  pred_group = {}
  truth_group = {}
  key_to_reviews = {}
  for i in xrange(len(votes)):
    voter = votes[i]['voter']
    product = reviews[votes[i]['review']]['product']
    key = (voter, product)
    if key not in pred_group:
      key_to_reviews[key] = []
      pred_group[key] = []
      truth_group[key] =[]
    pred_group[key].append(pred[i])
    truth_group[key].append(votes[i]['vote'])
    key_to_reviews[key].append(votes[i]['review'])
  scores = [] 
  sum_scores = 0
  max_score = 0
  max_rank = None
  min_score = float('inf')
  min_rank = None
  for key in pred_group:
    score = calculate_err(pred_group[key], truth_group[key])
    sum_scores += score
  score = sum_scores / len(pred_group)
  print >> output, 'ERR: %f' % score
  return score 



def main():
  """ Main function.

      Args:
        None.

      Returns:
        None. Results are output to 
      /out/res/<predictor>-<set_type>-<set_index>.dat file and to stdout.
  """
  predictor, set_type, rep = parse_args()
  rmse = []
  ndcg = []
  err = []
  output = open('out/res/%s-%s.dat' % (predictor, set_type), 'w')
  for i in xrange(NUM_SETS):
    rmse_sum = 0
    ndcg_sum = array([0] * RANK_SIZE)
    err_sum = 0
    repetitions = REP if rep else 1
    print >> output, 'Results on Set Type %d' % (i+1)
    for j in xrange(repetitions):
      votes, pred, reviews = load_data(predictor, set_type, i, j)
      rmse_sum += evaluate_regression(pred, votes, output)
      ndcg_sum = ndcg_sum + array(evaluate_ranking(pred, votes, reviews, output))
      err_sum += evaluate_ranking_err(pred, votes, reviews, output)
    rmse.append(rmse_sum / repetitions)
    ndcg.append(ndcg_sum / repetitions)
    err.append(err_sum / repetitions)
    print >> output, '-----'
  if rep:
    for i in xrange(NUM_SETS):
      print >> output, 'RMSE on set %d: %f' % (i+1, rmse[i])
      for j in xrange(len(ndcg[i])):
        print >> output, 'nDCG@%d on set %d: %f' % (j+1, i+1, ndcg[i][j])
      print >> output, 'ERR on set %d: %f' % (i+1, err[i])
  mean_rmse = mean(rmse)
  sd_rmse = std(rmse, ddof=1)
  e = t.ppf(CONF_QT, len(rmse)-1) * sd_rmse / sqrt(len(rmse)) 
  print 'IC of RMSE: (%f, %f) = %f : %f' % (mean_rmse - e, mean_rmse + e,
      mean_rmse, e)
  print >> output, rmse 
  print >> output, 'IC of RMSE: (%f, %f) = %f : %f' % (mean_rmse - e, mean_rmse
      + e, mean_rmse, e)
  for i in xrange(RANK_SIZE):
    scores = [score[i] for score in ndcg] 
    mean_ndcg = mean(scores)
    sd_ndcg = std(scores, ddof=1)
    e = t.ppf(CONF_QT, len(scores)-1) * sd_ndcg / sqrt(len(scores)) 
    print 'IC of nDCG@%d: (%f, %f) = %f : %f' % (i+1, mean_ndcg - e,
        mean_ndcg + e, mean_ndcg, e)
    if i + 1 == RANK_SIZE:
      print scores
      print >> output, scores
    print >> output, 'IC of nDCG@%d: (%f, %f) = %f : %f' % (i+1, mean_ndcg - e, 
        mean_ndcg + e, mean_ndcg, e)
  mean_err = mean(err)
  sd_err = std(err, ddof=1)
  e = t.ppf(CONF_QT, len(err)-1) * sd_err / sqrt(len(err)) 
  print 'IC of ERR: (%f, %f) = %f : %f' % (mean_err - e, mean_err + e, mean_err,
      e)
  print err 
  print >> output, err
  print >> output, 'IC of ERR: (%f, %f) = %f : %f' % (mean_err - e, mean_err +
      e, mean_err, e)
  output.close()

if __name__ == '__main__':
  main()
