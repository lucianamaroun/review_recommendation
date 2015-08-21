""" MF Module
    ---------

    Implementation of Matrix Factorization for prediction of helpfulness votes.
    Voter and review are the dimensions considered in the model and modeled as
    latent vectors.

    Usage:
    $ python -m betf.mf
"""


from math import sqrt

from numpy import nan, isnan
from numpy.random import random

from cap.aux import sigmoid, sigmoid_der1
from src.modeling.modeling import _SAMPLE_RATIO


K = 2

_ITER = 1000    # number of iterations of stochastic gradient descent
_ALPHA = 0.01   # learning rate
_BETA = 0.01    # regularization factor


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
    for _ in xrange(_ITER):
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
        pred.append(self.U[u,:].dot(self.R[r,:].T))
      else:
        pred.append(nan)
    return pred


if __name__ == "__main__":
  import pickle
  print 'Reading pickles'
 # _, _, _, train, test = model()
 # pickle.dump(train, open('pkl/cap_train%f.pkl' % _SAMPLE_RATIO, 'w'))
 # pickle.dump(test, open('pkl/cap_test%f.pkl' % _SAMPLE_RATIO, 'w'))
  train = pickle.load(open('pkl/cap_train%f.pkl' % _SAMPLE_RATIO, 'r'))
  test = pickle.load(open('pkl/cap_test%f.pkl' % _SAMPLE_RATIO, 'r'))
  overall_avg = float(sum([float(v['vote']) / 5.0 for v in train])) \
      / len(train)
  
  print 'Fitting Model'
  model = MF_Model()
  model.fit(train)

  print 'Calculate Predictions'
  pred = model.predict(train)
   
  print "TRAINING ERROR"
  sse = sum([(overall_avg if isnan(pred[i]) else 
      pred[i] - train[i]['vote'] / 5.0) ** 2 for i in xrange(len(train))])
  rmse = sqrt(sse/len(train))
  print 'RMSE: %s' % rmse
  
  pred = model.predict(test) 
  print "TESTING ERROR"
  sse = sum([(overall_avg if isnan(pred[i]) else 
      pred[i] - test[i]['vote'] / 5.0) ** 2 for i in xrange(len(test))])
  rmse = sqrt(sse/len(test))
  print 'RMSE: %s' % rmse

