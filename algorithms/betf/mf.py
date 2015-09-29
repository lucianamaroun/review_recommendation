""" MF Module
    ---------

    Implementation of Matrix Factorization for prediction of helpfulness votes.
    Voter and review are the dimensions considered in the model and modeled as
    latent vectors.

    Usage:
      $ python -m methods.betf.mf [-s <sample_size>] [-k <k>]
    where <sample_size> is a float with the fraction of the sample and K is an
    integer with the number of latent factor dimensions.
"""


from math import sqrt
from sys import argv, exit

from numpy import nan, isnan
from numpy.random import random
from pickle import load

from util.aux import sigmoid, sigmoid_der1
from evaluation.metrics import calculate_rmse, calculare_ndcg


K = 2
_ITER = 1000    # number of iterations of stochastic gradient descent
_ALPHA = 1      # starting learning rate (MOGHADDAM, with update)
_BETA = 0.1     # regularization factor (MOGHADDAM)
_SAMPLE = 0.05 
_TOL = 1e-6
_OUTPUT_DIR = 'out/pred/'


def load_args():
  """ Loads arguments.

      Args:
        None.

      Returns:
        A float with the sample size, an integer with dimension K.
  """
  i = 1
  while i < len(argv): 
    if argv[i] == '-s':
      global _SAMPLE
      _SAMPLE = float(argv[i+1])
    elif argv[i] == '-k':
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

  def fit(self, votes):
    """ Fits a MF model given training set (votes).

        Args:
          vote: list of votes, represented as dictionaries (training set).
        
        Returns:
          None. Instance fields are updated.
    """
    self._initialize_matrices(votes)
    previous = float('inf')
    for it in xrange(_ITER):
      _ALPHA = 1.0 / sqrt(it+1)
      for vote in votes:
        u = self.user_map[vote['voter']]
        r = self.review_map[vote['review']]
        dot = self.U[u,:].dot(self.R[r,:].T)
        error = float(vote['vote']) / 5.0 - sigmoid(dot) # normalized in (0,1)
        der_sig = sigmoid_der1(dot)
        for i in xrange(K):
          self.U[u,i] += _ALPHA * (2 * error * der_sig * self.R[r,i] - \
              _BETA * self.U[u,i])
          self.R[r,i] += _ALPHA * (2 * error * der_sig * self.U[u,i] - \
              _BETA * self.R[r,i])
      value = 0.0
      for vote in votes:
        u = self.user_map[vote['voter']]
        r = self.review_map[vote['review']]
        dot = self.U[u,:].dot(self.R[r,:].T)
        value += (vote['vote'] / 5.0 - sigmoid(dot)) ** 2
        for k in xrange(K):
          value += _BETA * (self.U[u,k] ** 2 + self.R[r,k] ** 2)
      if abs(previous - value) < _TOL:
        print 'Break'
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
        pred.append(sigmoid(self.U[u,:].dot(self.R[r,:].T)))
      else:
        pred.append(nan)
    return pred


if __name__ == '__main__':
  load_args()
  
  print 'Reading pickles'
  train = load(open('pkl/train%f.pkl' % _SAMPLE, 'r'))
  test = load(open('pkl/test%f.pkl' % _SAMPLE, 'r'))
  overall_avg = float(sum([float(v['vote']) for v in train])) / len(train)
  
  print 'Fitting Model'
  model = MF_Model()
  model.fit(train)

  print 'Calculating Predictions'
  pred = model.predict(train)
   
  print 'TRAINING ERROR'
  pred = [p * 5.0 for p in pred]
  truth = [v['vote'] for v in train]
  rmse = calculate_rmse(pred, truth) 
  print 'RMSE: %s' % rmse
  for i in xrange(5, 21, 5):
    score = calculate_ndcg(pred, truth, i)
    print 'NDCG@%d: %f' % (i, score)
    print >> output, 'NDCG@%d: %f' % (i, score)
  
  print 'Outputing Prediction'
  pred = model.predict(test) 
  output = open('%s/mf%f.dat' % (_OUTPUT_DIR, _SAMPLE), 'w')
  for p in pred:
    print >> output, overall_avg if isnan(p) else p
  output.close()

