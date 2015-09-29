""" Processing Output Module
    ------------------------

    Process output of a prediction and evaluates using RMSE and nDCG.

    Usage:
      python -m evaluation.process_output <sample_size> <predictor> <set_type>
    where <sample_size> is a float with the size of sample to use,
    <predictor> is a string with the name of the predictor method, and
    <set_type> is 'val' if validation set is to be used or 'test' if test set is
    desired.
"""


from math import sqrt
from sys import argv, exit
from pickle import load
from os.path import isfile

from evaluation.metrics import calculate_rmse, calculate_ndcg


def parse_args():
  """ Parses and validates command line arguments.

      Args:
        None.

      Returns:
        A float with sample size and a string with predictor name. 
  """
  if len(argv) != 4:
    print ('Usage: python -m script.evaluation <sample_size_float> '
        '<predictor_name_string> <set_type_string>')
    exit()
  try:
    sample = float(argv[1])
  except:
    print 'Sample size has to be a float'
  predictor = argv[2]
  if not isfile('out/pred/%s%.2f.dat' % (predictor, sample * 100)):
    print 'There is no file with predicted values of %s for %f sample' % \
        (predictor, sample)
    exit()
  set_type = argv[3]
  if set_type != 'val' and set_type != 'test':
    print 'Set type has to be either \'val\' or \'test\''
    exit()
  return sample, predictor, set_type


def load_data(sample, predictor, set_type):
  """ Loads data of predicted and true values to evaluate. 

      Args:
        sample: a float with sample size.
        predictor: a string with predictor name.
        set_type: string with the set type to be used, 'val' for validation set
      and 'test' for test set.

      Returns:
        A float with sample size and a string with predictor name. 
  """
  if set_type == 'val': 
    votes = load(open('out/pkl/validation%.2f.pkl' % (sample * 100), 'r'))
    predfile = open('out/val/%s%.2f.dat' % (predictor, sample * 100), 'r')
  else:
    votes = load(open('out/pkl/test%.2f.pkl' % (sample * 100), 'r'))
    predfile = open('out/pred/%s%.2f.dat' % (predictor, sample * 100), 'r')
  pred = [float(line.strip()) for line in predfile]
  predfile.close()
  return votes, pred


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


def evaluate_ranking(pred, votes, output):
  """ Evaluates predicted values using nDCG@K, with K ranging from 1 to 20,
      a ranking metric.

      Args:
        pred: a list of floats with predicted values.
        votes: a list of votes, represented as dictionaries, belonging to
      votes set.
        output: a file object to print output on.

      Returns:
        None. The result is printed on output file and stdout.
  """
  pred_group = {}
  truth_group = {}
  for i in xrange(len(votes)):
    voter = votes[i]['voter']
    if voter not in pred_group:
      pred_group[voter] = []
      truth_group[voter] =[]
    pred_group[voter].append(pred[i])
    truth_group[voter].append(votes[i]['vote'])
  for i in xrange(1, 11):
    score_sum = 0.0
    for key in pred_group:
      score_sum += calculate_ndcg(pred_group[key], truth_group[key], i)
    score = score_sum / len(pred_group)
    if i % 5 == 0:
      print 'NDCG@%d: %f (grouped V)' % (i, score)
    print >> output, 'NDCG@%d: %f (grouped V)' % (i, score)


def main():
  """ Main function.

      Args:
        None.

      Returns:
        None. Results are output to /out/res/<predictor><sample>.dat file and to
      stdout.
  """
  sample, predictor, set_type = parse_args()
  votes, pred = load_data(sample, predictor, set_type)
  output = open('out/res/%s%.2f.dat' % (predictor, sample * 100), 'w')
  evaluate_regression(pred, votes, output)
  evaluate_ranking(pred, votes, output)
  output.close()

if __name__ == '__main__':
  main()
