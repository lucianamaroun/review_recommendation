""" Mean-based Prediction
    ---------------------

    Implements simple predictors using mean statistics from test set.

    Usage:
      $ python -m methods.regression.prediction [-p <predictor>]
    where <predictor> is in the set [om, rm, am, vm]
"""


from sys import argv, exit

from pickle import load

from algorithms.const import NUM_SETS, RANK_SIZE
from evaluation.metrics import calculate_rmse, calculate_ndcg


_PREDICTORS = ['om', 'rm', 'am', 'vm']
_PRED = 'om'
_PKL_DIR = 'out/pkl'
_OUTPUT_DIR = 'out/pred'


def load_args():
  """ Loads arguments.

      Args:
        None.

      Returns:
        None. Module variables are initialized. 
  """
  i = 1
  while i < len(argv): 
    if argv[i] == '-p' and argv[i+1] in _PREDICTORS:
      global _PRED 
      _PRED = argv[i+1]
    else:
      print ('Usage: python -m methods.regression.prediction [-s <sample_size>]'
          '[-p <predictor>], <predictor> is in the set [om, rm, am, vm]')
      exit()
    i = i + 2


def compute_overall_mean(votes):
  """ Computes the mean of all helpfulness votes as a constant prediction value.

      Args:
        votes: a list of votes to learn from, calculating the mean.

      Returns:
        A function which maps from a dictionary vote to a prediction, a real
      value.
  """
  overall_mean = 0
  for vote in votes:
    truth = vote['vote']
    overall_mean += truth 
  overall_mean /= float(len(votes))
  return lambda vote: overall_mean


def compute_review_mean(votes):
  """ Computes the mean of all helpfulness votes of each review as predicted
      value of new votes given to the review. When the review is unknown, that
      is, is not in test set and no average is available, then overall mean is
      used instead for prediction.

      Args:
        votes: a list of votes to learn from, calculating the mean of each
      review.

      Returns:
        A function which maps from a dictionary vote to a prediction, a real
      value.
  """
  overall_mean = 0
  review_mean = {}
  for vote in votes:
    truth = vote['vote']
    overall_mean += truth 
    review = vote['review']
    if review not in review_mean:
      review_mean[review] = {}
      review_mean[review]['sum'] = 0
      review_mean[review]['count'] = 0
    review_mean[review]['sum'] += truth
    review_mean[review]['count'] += 1 
  overall_mean /= float(len(votes))
  for review in review_mean:
    total, count = review_mean[review]['sum'], review_mean[review]['count']
    review_mean[review] = float(total) / count
  return lambda vote: review_mean[vote['review']] if vote['review'] in \
      review_mean else overall_mean


def compute_author_mean(votes):
  """ Computes the mean of all helpfulness votes received by each author as
      prediction of new votes given to a review written by the same author. When 
      the author is unknown, that is, is not in test set and no average is
      available, then overall mean is used instead for prediction.

      Args:
        votes: a list of votes to learn from, calculating the mean of each
      author.

      Returns:
        A function which maps from a dictionary vote to a prediction, a real
      value.
  """
  overall_mean = 0
  author_mean = {}
  for vote in votes:
    truth = vote['vote']
    overall_mean += truth 
    author = vote['author']
    if author not in author_mean:
      author_mean[author] = {}
      author_mean[author]['sum'] = 0
      author_mean[author]['count'] = 0
    author_mean[author]['sum'] += truth
    author_mean[author]['count'] += 1 
  overall_mean /= float(len(votes))
  for author in author_mean:
    total, count = author_mean[author]['sum'], author_mean[author]['count']
    author_mean[author] = float(total) / count
  return lambda vote: author_mean[vote['author']] if vote['author'] in \
      author_mean else overall_mean


def compute_voter_mean(votes):
  """ Computes the mean of all helpfulness votes given by each voter as
      predicted value of new votes given by the same voter. When the voter is
      unknown, that is, is not in test set and no average is available, then
      overall mean is used instead for prediction.

      Args:
        votes: a list of votes to learn from, calculating the mean of each
      voter.

      Returns:
        A function which maps from a dictionary vote to a prediction, a real
      value.
  """
  overall_mean = 0
  voter_mean = {}
  for vote in votes:
    truth = vote['vote']
    overall_mean += truth 
    voter = vote['voter']
    if voter not in voter_mean:
      voter_mean[voter] = {}
      voter_mean[voter]['sum'] = 0
      voter_mean[voter]['count'] = 0
    voter_mean[voter]['sum'] += truth
    voter_mean[voter]['count'] += 1 
  overall_mean /= float(len(votes))
  for voter in voter_mean:
    total, count = voter_mean[voter]['sum'], voter_mean[voter]['count']
    voter_mean[voter] = float(total) / count
  return lambda vote: voter_mean[vote['voter']] if vote['voter'] in \
      voter_mean else overall_mean


def fit_predictor(votes):
  """ Fits a predictor to a set of training votes.

      Args:
        votes: the set of training votes.

      Returns:
        A function which maps a vote dictionary to a real value.
  """
  if _PRED == 'om':
    return compute_overall_mean(votes)
  elif _PRED == 'rm':
    return compute_review_mean(votes)
  elif _PRED == 'am':
    return compute_author_mean(votes)
  else:
    return compute_voter_mean(votes)


def main():
  """ Main module. Fits a mean-based predictor on training set, loaded from
      pickle, and predict votes for a test set, loaded from pickle, outputing to
      a file with predicted values and displaying training performance on
      stdout.

      Args:
        None.

      Returns:
        None.
  """
  load_args()

  for i in xrange(NUM_SETS):
    train = load(open('%s/train-%d.pkl' % (_PKL_DIR, i), 'r'))
    test = load(open('%s/test-%d.pkl' % (_PKL_DIR, i), 'r'))
    reviews = load(open('%s/reviews-%d.pkl'% (_PKL_DIR, i), 'r'))
    predictor = fit_predictor(train)
    pred = [predictor(v) for v in train]
    truth = [v['vote'] for v in train]
    print 'TRAINING ERROR'
    print '-- RMSE: %f' % calculate_rmse(pred, truth) 
    pred_group = {}
    truth_group = {}
    for vote in train:
      voter = vote['voter']
      product = reviews[vote['review']]['product']
      key = (voter, product)
      if key not in pred_group:
        pred_group[key] = []
        truth_group[key] =[]
      pred_group[key].append(pred[i])
      truth_group[key].append(truth[i])
    score_sum = 0.0
    for key in pred_group:
      score_sum += calculate_ndcg(pred_group[key], truth_group[key], RANK_SIZE)
    score = score_sum / len(pred_group)
    print '-- nDCG@%d: %f' % (RANK_SIZE, score)
    output = open('%s/%s-%d-0.dat' % (_OUTPUT_DIR, _PRED, i), 'w')
    for v in test:
      print >> output, predictor(v)
    output.close()
  


if __name__ == '__main__':
  main()
