""" Processing Output Module
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
from numpy import mean, std, array
from os.path import isfile
from pickle import load
from scipy.stats import t 
from sys import argv, exit

from algo.const import NUM_SETS, CONF_QT, RANK_SIZE, REP
from perf.metrics import calculate_rmse, calculate_ndcg


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
    predfile = open('out/pred/%s-%d-%d.dat'% (predictor, index, rep), 'r')
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
        output: a file object to print output on.

      Returns:
        None. The result is printed on output file and stdout.
  """
  truth = [v['vote'] for v in votes]
  rmse = calculate_rmse(pred, truth) 
  print "RMSE: %f" % rmse
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
  for i in xrange(len(votes)):
    voter = votes[i]['voter']
    product = reviews[votes[i]['review']]['product']
    key = (voter, product)
    if key not in pred_group:
      pred_group[key] = []
      truth_group[key] =[]
    pred_group[key].append(pred[i])
    truth_group[key].append(votes[i]['vote'])
  scores = [] 
  for i in xrange(1, RANK_SIZE+1):
    sum_scores = 0
    for key in pred_group:
      sum_scores += calculate_ndcg(pred_group[key], truth_group[key], i)
    score = sum_scores / len(pred_group)
    if i % RANK_SIZE == 0:
      print 'NDCG@%d: %f' % (i, score)
    print >> output, 'NDCG@%d: %f' % (i, score)
    scores.append(score)
  return scores 

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
  for i in xrange(NUM_SETS):
    rmse_sum = 0
    ndcg_sum = array([0] * RANK_SIZE)
    repetitions = REP if rep else 1
    for j in xrange(repetitions):
      votes, pred, reviews = load_data(predictor, set_type, i, j)
      output = open('out/res/%s-%s-%d-%d.dat' % (predictor, set_type, i, j), 'w')
      rmse_sum += evaluate_regression(pred, votes, output)
      ndcg_sum = ndcg_sum + array(evaluate_ranking(pred, votes, reviews, output))
      output.close()
    rmse.append(rmse_sum / repetitions)
    ndcg.append(ndcg_sum / repetitions)

  output = open('out/res/%s-%s.dat' % (predictor, set_type), 'w')
  mean_rmse = mean(rmse)
  sd_rmse = std(rmse, ddof=1)
  err = t.cdf(CONF_QT, len(rmse)-1) * sd_rmse / sqrt(len(rmse)-1) 
  print 'IC of RMSE: (%f, %f)' % (mean_rmse - err, mean_rmse + err)
  print >> output, 'IC of RMSE: (%f, %f)' % (mean_rmse - err, mean_rmse + err)
  for i in xrange(RANK_SIZE):
    scores = [score[i] for score in ndcg] 
    mean_ndcg = mean(scores)
    sd_ndcg = std(scores, ddof=1)
    err = t.cdf(CONF_QT, len(scores)-1) * sd_ndcg / sqrt(len(scores)-1) 
    print 'IC of nDCG@%d: (%f, %f)' % (i+1, mean_ndcg - err, mean_ndcg + err)
    print >> output, 'IC of nDCG@%d: (%f, %f)' % (i+1, mean_ndcg - err, mean_ndcg + err)
  output.close()

if __name__ == '__main__':
  main()
