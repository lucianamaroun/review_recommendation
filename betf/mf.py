from numpy import nan
from numpy.random import random

from cap.aux import sigmoid, sigmoid_der1

K = 2

_ITER = 1000    # number of iterations of stochastic gradient descent
_ALPHA = 0.01   # learning rate
_BETA = 0.01    # regularization factor


class MF_Model(object):

  def __init__(self):
    self.U = None # Matrix of user (voter( latent arrays (N_v, K)
    self.R = None # Matrix of review latent arrays (N_r, K)
    self.user_map = None # Map from user ids to matrix indices (lines)
    self.review_map = None # Map from review ids to matrix indices (lines)

  def initialize_matrices(self, votes):
    users = sorted(set([vote['voter'] for vote in votes]))
    reviews = sorted(set([vote['review'] for vote in votes]))
    self.user_map = {u:i for i, u in enumerate(users)}
    self.review_map = {r:i for i, r in enumerate(reviews)}
    self.U = random((len(users), K))
    self.R = random((len(reviews), K))

  def fit(self, votes):
    self.initialize_matrices(votes)
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
    pred = []
    for vote in votes:
      u = self.user_map[vote['voter']] if vote['voter'] in user_map else -1
      r = self.review_map[vote['review']] if vote['review'] in user_map else -1
      if u != -1 and r != -1:
        pred.append(U[u,:].dot(R[r,:].T)[0,0])
      else:
        pred.append(nan)

