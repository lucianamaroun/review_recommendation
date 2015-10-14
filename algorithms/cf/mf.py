""" MF Module
    ---------

    Implementation of Matrix Factorization for prediction of helpfulness votes.
    Voter and review are the dimensions considered in the model and modeled as
    latent vectors.

    Usage:
      $ python -m methods.betf.mf [-k <k>]
    where <sample_size> is a float with the fraction of the sample and K is an
    integer with the number of latent factor dimensions.
"""


from math import sqrt
from sys import argv, exit

from numpy import nan, isnan
from numpy.random import random
from pickle import load

from algorithms.const import NUM_SETS, RANK_SIZE, REP 
from evaluation.metrics import calculate_rmse, calculate_ndcg


K = 5
_ITER = 1000      # number of iterations of stochastic gradient descent
_ALPHA = 0.01     # starting learning rate (MOGHADDAM, with update)
_BETA = 0.01      # regularization factor (MOGHADDAM)
_TOL = 1e-6
_VAL_DIR = 'out/val'
_OUTPUT_DIR = 'out/pred'
_PKL_DIR = 'out/pkl'


def load_args():
  """ Loads arguments.

      Args:
        None.

      Returns:
        A float with the sample size, an integer with dimension K.
  """
  i = 1
  while i < len(argv): 
    if argv[i] == '-k':
      global K
      K = int(argv[i+1])
    else:
      print 'Usage: python -m methods.betf.mf [-s <sample_size>] [-k <k>]'
      exit()
    i = i + 2


class MF_Model(object):
  """ Class implementing a Matrix Factorization Model. """

  def __init__(self):
    """ Discriminates existing attributes, initilizing all to None.

        Args:
          None.

        Returns:
          None.
    """
    self.U = None # Matrix of user (voter( latent arrays (N_v, K)
    self.R = None # Matrix of review latent arrays (N_r, K)
    self.user_map = None # Map from user ids to matrix indices (lines)
    self.review_map = None # Map from review ids to matrix indices (lines)
    self.review_bias = none
    self.user_bias = none
    self.overall_mean = None

  def _initialize_matrices(self, votes):
    """ Initializes matrices and mappings given votes. Each entity id is mapped
        to an index in a dimension of  the matrix.

        Args:
          votes: list of votes (training set).
        
        Returns:
          None. Instance fields are updated.
    """
    users = sorted(set([vote['voter'] for vote in votes]))
    reviews = sorted(set([vote['review'] for vote in votes]))
    self.user_map = {u:i for i, u in enumerate(users)}
    self.review_map = {r:i for i, r in enumerate(reviews)}
    self.U = random((len(users), K))
    self.R = random((len(reviews), K))

  def _calculate_bias(self, votes):
    self.user_bias = {}
    user_count = {}
    self.review_bias = {}
    review_count = {}
    self.overall_mean = 0
    count = 0
    for vote in votes:
      self.overall_mean += vote['vote']
      count += 1
    self.overall_mean /= float(count)
    for vote in votes:
      user = vote['voter']
      if user not in self.user_bias:
        self.user_bias[user] = 0
        user_count[user] = 0
      self.user_bias[user] += (vote['vote'] - self.overall_mean)
      user_count[user] += 1
    for user in self.user_bias:
      self.user_bias[user] /= float(user_count[user])
    for vote in votes:
      user = vote['voter']
      review = vote['review']
      if review not in self.review_bias:
        self.review_bias[review] = 0
        review_count[review] = 0
      self.review_bias[review] += (vote['vote'] - self.overall_mean -
          self.user_bias[user]) 
      review_count[review] += 1
    for review in self.review_bias:
      self.review_bias[review] /= float(review_count[review])

  def fit(self, votes):
    """ Fits a MF model given training set (votes).

        Args:
          vote: list of votes, represented as dictionaries (training set).
        
        Returns:
          None. Instance fields are updated.
    """
    self._initialize_matrices(votes)
    self._calculate_bias(votes)
    previous = float('inf')
    for it in xrange(_ITER):
      for vote in votes:
        u = self.user_map[vote['voter']]
        r = self.review_map[vote['review']]
        dot = self.overall_mean + self.user_bias[vote['voter']] + \
            self.review_bias[vote['review']] + self.U[u,:].dot(self.R[r,:].T)
        error = float(vote['vote']) - dot
        self.user_bias[vote['voter']] += _ALPHA * 2 * (error - _BETA *
            self.user_bias[vote['voter']])
        self.review_bias[vote['review']] += _ALPHA * 2 * (error - _BETA *
            self.review_bias[vote['review']])
        for i in xrange(K):
          self.U[u,i] += _ALPHA * (2 * error * self.R[r,i] - \
              _BETA * self.U[u,i])
          self.R[r,i] += _ALPHA * (2 * error * self.U[u,i] - \
              _BETA * self.R[r,i])
      value = 0.0
      for vote in votes:
        u = self.user_map[vote['voter']]
        r = self.review_map[vote['review']]
        dot = self.overall_mean + self.user_bias[vote['voter']] + \
            self.review_bias[vote['review']] + self.U[u,:].dot(self.R[r,:].T)
        value += (vote['vote'] - dot) ** 2
        for k in xrange(K):
          value += _BETA * (self.U[u,k] ** 2 + self.R[r,k] ** 2)
      if abs(previous - value) < _TOL:
        print 'Convergence'
        break
      previous = value
  
  def predict(self, votes):
    """ Predicts a set of vote examples using previous fitted model.

        Args:
          votes: list of dictionaries, representing votes, to predict
        helpfulness vote value.

        Returns:
          A list of floats with predicted vote values.
    """
    pred = []
    for vote in votes:
      u = self.user_map[vote['voter']] if vote['voter'] in self.user_map else -1
      r = self.review_map[vote['review']] if vote['review'] in self.review_map else -1
      if u != -1 and r != -1:
        dot = self.overall_mean + self.user_bias[vote['voter']] + \
            self.review_bias[vote['review']] + self.U[u,:].dot(self.R[r,:].T) 
        pred.append(dot)
      else:
        pred.append(nan)
    return pred


if __name__ == '__main__':
  load_args()
  
  for i in xrange(NUM_SETS):
    print 'Reading pickles'
    train = load(open('%s/train-%d.pkl' % (_PKL_DIR, i), 'r'))
    val = load(open('%s/validation-%d.pkl' % (_PKL_DIR, i), 'r'))
    test = load(open('%s/test-%d.pkl' % (_PKL_DIR, i), 'r'))
    reviews = load(open('%s/reviews-%d.pkl' % (_PKL_DIR, i), 'r'))
    truth = [v['vote'] for v in train]
    overall_avg = float(sum(truth)) / len(train)
    
    for j in xrange(REP):
      print 'Fitting Model'
      model = MF_Model()
      model.fit(train)

      print 'Calculating Predictions'
      pred = model.predict(train)
       
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
        ndcg = calculate_ndcg(pred_group[key], truth_group[key], RANK_SIZE)
        score_sum += ndcg
      score = score_sum / len(pred_group)
      print '-- nDCG@%d: %f' % (RANK_SIZE, score)
      
      pred = model.predict(val) 
      print 'Outputting validation prediction'
      output = open('%s/mf-%d-%d.dat' % (_VAL_DIR, i, j), 'w')
      for p in pred:
        print >> output, overall_avg if isnan(p) else p
      output.close()
      
      pred = model.predict(test) 
      print 'Outputting testing prediction'
      output = open('%s/mf-%d-%d.dat' % (_OUTPUT_DIR, i, j), 'w')
      for p in pred:
        print >> output, overall_avg if isnan(p) else p
      output.close()