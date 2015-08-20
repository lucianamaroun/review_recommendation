from numpy import nan
from numpy.random import random

from cap.aux import sigmoid, sigmoid_der1

K = 2

_ITER = 1000    # number of iterations of stochastic gradient descent
_ALPHA = 0.01   # learning rate
_BETA = 0.01    # regularization factor



class TF_Model(object):

  def __init__(self):
    self.V = None # Matrix of user (voter) latent arrays (N_v, K)
    self.A = None # Matrix of user (author) latent arrays (N_a, K)
    self.P = None # Matrix of product latent arrays (N_p, K)
    self.S = None # Central tensor
    self.voter_map = None 
    self.product_map = None 
    self.author_map = None 

  def initialize_matrices(self, votes):
    voters = sorted(set([vote['voter'] for vote in votes]))
    products = sorted(set([vote['product'] for vote in votes]))
    authors = sorted(set([vote['user'] for vote in votes]))
    self.voter_map = {u:i for i, u in enumerate(users)}
    self.product_map = {p:i for i, p in enumerate(products)}
    self.author_map = {a:i for i, a in enumerate(authors)}
    self.U = random((len(users), K))
    self.R = random((len(reviews), K))
    self.A = random((len(reviews), K))
    self.S = random((K, K, K))

  def tensor_dot(self, v, a, p):
    dot = 0
    for x in xrange(K):
      for y in xrange(K):
        for z in xrange(K):
          dot += self.S[x,y,z] * self.V[v,x] * self.A[a,y] * self.P[p,z]
    return dot
  
  def tensor_dot_der_v(self, a, p, i):
    dot = 0
    for y in xrange(K):
      for z in xrange(K):
        dot += self.S[i,y,z] * self.A[a,y] * self.P[p,z]
    return dot
  
  def tensor_dot_der_a(self, v, p, i):
    dot = 0
    for x in xrange(K):
      for z in xrange(K):
        dot += self.S[x,i,z] * self.V[v,x] * self.P[p,z]
    return dot
  
  def tensor_dot_der_a(self, v, a, i):
    dot = 0
    for x in xrange(K):
      for y in xrange(K):
        dot += self.S[x,y,i] * self.V[v,x] * self.A[a,y]
    return dot
  
  def fit(self, votes):
    self.initialize_matrices(votes)
    for _ in xrange(_ITER):
      for vote in votes:
        v = self.user_map[vote['voter']]
        a = self.author_map[vote['user']]
        p = self.review_map[vote['product']]
        dot = self.tensor_dot(v, a, p)
        error = float(vote['vote']) / 5.0 - sigmoid(dot) # normalized in (0,1)
        der_sig = sigmoid_der1(dot)
        for i in xrange(K):
          self.V[v,i] += _ALPHA * (2 * error * der_sig * tensor_dot_der_v(a, p,
              i) - _BETA * self.V[v,i])
          self.A[a,i] += _ALPHA * (2 * error * der_sig * tensor_dot_der_a(v, p,
              i) - _BETA * self.A[a,i])
          self.P[p,i] += _ALPHA * (2 * error * der_sig * tensor_dot_der_p(a, v,
              i) - _BETA * self.P[p,i])
          for j in xrange(K):
            for k in xrange(K):
            self.S[i,j,k] += _ALPHA * (2 * error * der_sig * self.V[v,i] *
              self.A[a,j] * self.P[p,k] - _BETA * self.S[i,j,k])
           
  def predict(self, votes):
    pred = []
    for vote in votes:
      u = self.user_map[vote['voter']] if vote['voter'] in user_map else -1
      r = self.review_map[vote['review']] if vote['review'] in user_map else -1
      if u != -1 and r != -1:
        pred.append(U[u,:].dot(R[r,:].T)[0,0])
      else:
        pred.append(nan)

