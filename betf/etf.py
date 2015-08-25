""" ETF Module
    ---------

    Implementation of Tensor Factorization for prediction of helpfulness votes.
    Voter, author and product are the dimensions considered in the model and
    modeled as latent vectors.

    Usage:
    $ python -m betf.etf
"""


from math import sqrt

from numpy import nan, isnan, tensordot
from numpy.random import random

from cap.aux import sigmoid, sigmoid_der1
from src.modeling.modeling import model, _SAMPLE_RATIO


K = 2
_ITER = 1000    # number of iterations of stochastic gradient descent
_ALPHA = 0.01   # learning rate
_BETA = 0.01    # regularization factor
SAMPLE = 0.001


class ETF_Model(object):
  """ Class implementing a Matrix Factorization Model. """

  def __init__(self):
    """ Discriminates existing attributes, initilizing all to None.

        Args:
          None.

        Returns:
          None.
    """
    self.V = None # Matrix of user (voter) latent arrays (N_v, K)
    self.A = None # Matrix of user (author) latent arrays (N_a, K)
    self.P = None # Matrix of product latent arrays (N_p, K)
    self.S = None # Central tensor
    self.voter_map = None 
    self.author_map = None 
    self.product_map = None 

  def _initialize_matrices(self, votes, reviews):
    """ Initializes matrices and mappings given votes. Each entity id is mapped
        to an index in a dimension of  the matrix.

        Args:
          votes: list of votes (training set).
        
        Returns:
          None. Instance fields are updated.
    """
    voters = sorted(set([vote['voter'] for vote in votes]))
    authors = sorted(set([vote['reviewer'] for vote in votes]) \
        .union(set([review['user'] for review in reviews.itervalues()])))
    products = sorted(set([reviews[v['review']]['product'] for v in votes]) \
        .union(set([review['product'] for review in reviews.itervalues()])))
    self.voter_map = {u:i for i, u in enumerate(voters)}
    self.author_map = {a:i for i, a in enumerate(authors)}
    self.product_map = {p:i for i, p in enumerate(products)}
    self.V = random((len(voters), K))
    self.A = random((len(authors), K))
    self.P = random((len(products), K))
    self.S = random((K, K, K))

  def tensor_dot(self, v, a, p):
    """ Performs a tensor dot of three vectors and the central tensor.
            
        Args:
          v: index of vector in V matrix.
          a: index of vector in A matrix.
          p: index of vector in P matrix.

        Returns:
          A float, the dot value.
    """    
    dot = 0.0
    for x in xrange(K):
      for y in xrange(K):
        for z in xrange(K):
          dot += self.S[x,y,z] * self.V[v,x] * self.A[a,y] * self.P[p,z]
    return dot
  
  def tensor_dot_der_v(self, a, p):
    """ Computes the derivative of the tensor dot relative to 'v' variable.

        Args:
          a: index of vector in A matrix.
          p: index of vector in P matrix.
    
        Return:
          A k-array with the derivative at each dimension of 'v'.
    """
    dot = 0.0
    for y in xrange(K):
      for z in xrange(K):
        dot += self.S[:,y,z] * self.A[a,y] * self.P[p,z]
    return dot
  
  def tensor_dot_der_a(self, v, p):
    """ Computes the derivative of the tensor dot relative to 'a' variable.

        Args:
          v: index of vector in V matrix.
          p: index of vector in P matrix.
    
        Return:
          A k-array with the derivative at each dimension of 'a'.
    """
    dot = 0.0
    for x in xrange(K):
      for z in xrange(K):
        dot += self.S[x,:,z] * self.V[v,x] * self.P[p,z]
    return dot
  
  def tensor_dot_der_p(self, v, a):
    """ Computes the derivative of the tensor dot relative to 'p' variable.

        Args:
          v: index of vector in V matrix.
          a: index of vector in A matrix.
    
        Return:
          A k-array with the derivative at each dimension of 'p'.
    """
    dot = 0
    for x in xrange(K):
      for y in xrange(K):
        dot += self.S[x,y,:] * self.V[v,x] * self.A[a,y]
    return dot

  def tensor_dot_der_s(self, v, a, p):
    """ Computes the derivative of the tensor dot relative to 's', the central
        tensor.

        Args:
          v: index of vector in V matrix.
          a: index of vector in A matrix.
          p: index of vector in P matrix.
    
        Return:
          A (k, k, k) tensor with the derivative at each cell of 's'.
    """
    temp = tensordot(self.V[v,:], self.A[a,:], 0)
    temp = tensordot(temp, self.P[p,:], 0)
    return temp
 
  def fit(self, votes, reviews):
    """ Fits a TF model given training set (votes).

        Args:
          vote: list of votes, represented as dictionaries (training set).
        
        Returns:
          None. Instance fields are updated.
    """
    self._initialize_matrices(votes, reviews)
    for _ in xrange(_ITER):
      for vote in votes:
        v = self.voter_map[vote['voter']]
        a = self.author_map[vote['reviewer']]
        p = self.product_map[reviews[vote['review']]['product']]
        dot = self.tensor_dot(v, a, p)
        error = float(vote['vote']) / 5.0 - sigmoid(dot) # normalized in (0,1)
        der_sig = sigmoid_der1(dot)
        self.V[v,:] += _ALPHA * (2 * error * der_sig * \
            self.tensor_dot_der_v(a, p) - _BETA * self.V[v,:])                                          
        self.A[a,:] += _ALPHA * (2 * error * der_sig * \
            self.tensor_dot_der_a(v, p) - _BETA * self.A[a,:])
        self.P[p,:] += _ALPHA * (2 * error * der_sig * \
            self.tensor_dot_der_p(v, a) - _BETA * self.P[p,:])                                          
        self.S += _ALPHA * (2 * error * der_sig * \
            self.tensor_dot_der_s(v, a, p) - _BETA * self.S) 
      for review in reviews.itervalues():
        a = self.author_map[review['user']]
        p = self.product_map[review['product']]
        dot = self.A[a,:].dot(self.P[p,:]) 
        error = float(review['rating']) / 5.0 - sigmoid(dot) # normalized in (0,1)
        der_sig = sigmoid_der1(dot)
        self.A[a,:] += _ALPHA * (2 * error * der_sig * self.P[p,] - \
            _BETA * self.A[a,:])
        self.P[p,:] += _ALPHA * (2 * error * der_sig * self.A[a,:] - \
            _BETA * self.P[p,:])                                          
      e = 0
      for vote in votes:
        v = self.voter_map[vote['voter']]
        a = self.author_map[vote['reviewer']]
        p = self.product_map[reviews[vote['review']]['product']]
        dot = self.tensor_dot(v, a, p)
        e += float(vote['vote']) / 5.0 - sigmoid(dot) # normalized in (0,1)
      for review in reviews.itervalues():
        a = self.author_map[review['user']]
        p = self.product_map[review['product']]
        dot = self.A[a,:].dot(self.P[p,:])
        e += float(review['rating']) / 5.0 - sigmoid(dot) # normalized in (0,1)
      if e < 1e-6:
        print "Break"
        break

  def predict(self, votes, reviews):
    """ Predicts a set of vote examples using previous fitted model.

        Args:
          votes: list of dictionaries, representing votes, to predict
        helpfulness vote value.

        Returns:
          A list of floats with predicted vote values.
    """
    pred = []
    for vote in votes:
      v = self.voter_map[vote['voter']] if vote['voter'] in self.voter_map else -1
      a = self.author_map[vote['reviewer']] if vote['reviewer'] in self.author_map else -1
      p = self.product_map[reviews[vote['review']]['product']] if reviews[vote['review']]['product'] in self.product_map else -1
      if v != -1 and a != -1 and p != -1:
        pred.append(self.tensor_dot(v, a, p))
      else:
        pred.append(nan)
    return pred


if __name__ == "__main__":
  import pickle
  print 'Reading pickles'
 # reviews, _, _, train, test = model()
 # pickle.dump(reviews, open('pkl/betf_reviews%f.pkl' % SAMPLE, 'w'))
 # pickle.dump(train, open('pkl/betf_train%f.pkl' % SAMPLE, 'w'))
 # pickle.dump(test, open('pkl/betf_test%f.pkl' % SAMPLE, 'w'))
  reviews = pickle.load(open('pkl/betf_reviews%f.pkl' % SAMPLE, 'r'))
  train = pickle.load(open('pkl/betf_train%f.pkl' % SAMPLE, 'r'))
  test = pickle.load(open('pkl/betf_test%f.pkl' % SAMPLE, 'r'))
  overall_avg = float(sum([float(v['vote']) / 5.0 for v in train])) \
      / len(train)
  
  print 'Fitting Model'
  model = ETF_Model()
  model.fit(train, reviews)

  print 'Calculate Predictions'
  pred = model.predict(train, reviews)
   
  print "TRAINING ERROR"
  sse = sum([(overall_avg if isnan(pred[i]) else 
      pred[i] - train[i]['vote'] / 5.0) ** 2 for i in xrange(len(train))])
  rmse = sqrt(sse/len(train))
  print 'RMSE: %s' % rmse
  
  pred = model.predict(test, reviews) 
  print "TESTING ERROR"
  sse = sum([(overall_avg if isnan(pred[i]) else 
      pred[i] - test[i]['vote'] / 5.0) ** 2 for i in xrange(len(test))])
  rmse = sqrt(sse/len(test))
  print 'RMSE: %s' % rmse

