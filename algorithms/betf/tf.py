""" TF Module
    ---------

    Implementation of Tensor Factorization for prediction of helpfulness votes.
    Voter, author and product are the dimensions considered in the model and
    modeled as latent vectors.

    Usage:
      $ python -m methods.betf.tf [-s <sample_size>] [-k <k>]
    where <sample_size> is a float with the fraction of the sample and K is an
    integer with the number of latent factor dimensions.
"""


from math import sqrt
from sys import argv, exit

from numpy import nan, isnan, tensordot
from numpy.random import random
from pickle import load

from util.aux import sigmoid, sigmoid_der1
from evaluation.metrics import calculate_rmse, calculare_ndcg


K = 2
_ITER = 1000    # number of iterations of stochastic gradient descent
_ALPHA = 1      # starting learning rate
_BETA = 0.1     # regularization factor
_SAMPLE = float(argv[1])
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
      print 'Usage: python -m methods.betf.tf [-s <sample_size>] [-k <k>]'
      exit()
    i = i + 2


class TF_Model(object):
  """ Class implementing a Tensor Factorization Model. """

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
    authors = sorted(set([vote['author'] for vote in votes]))
    products = sorted(set([reviews[vote['review']]['product'] for vote in votes]))
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
    previous = float('inf')
    for it in xrange(_ITER):
      _ALPHA = 1.0 / sqrt(it+1)
      for vote in votes:
        v = self.voter_map[vote['voter']]
        a = self.author_map[vote['author']]
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
      value = 0.0
      for vote in votes:
        v = self.voter_map[vote['voter']]
        a = self.author_map[vote['author']]
        p = self.product_map[reviews[vote['review']]['product']]
        dot = self.tensor_dot(v, a, p)
        value += (float(vote['vote']) / 5.0 - sigmoid(dot)) ** 2 # normalized in (0,1)
        for i in xrange(K):
          value += _BETA * (self.V[v,i] ** 2 + self.A[a,i] ** 2 + \
              self.P[p,i] ** 2)
          for j in xrange(K):
            for k in xrange(K):
              value += _BETA * (self.S[i,j,k] ** 2)
      if abs(previous - value) < _TOL:
        print 'Break'
        break
      previous = value

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
      v = self.voter_map[vote['voter']] if vote['voter'] in self.voter_map \
          else -1
      a = self.author_map[vote['author']] if vote['author'] in \
          self.author_map else -1
      p = self.product_map[reviews[vote['review']]['product']] if \
          reviews[vote['review']]['product'] in self.product_map else -1
      if v != -1 and a != -1 and p != -1:
        pred.append(sigmoid(self.tensor_dot(v, a, p)))
      else:
        pred.append(nan)
    return pred


if __name__ == '__main__':
  load_args()

  print 'Reading pickles'
  reviews = load(open('pkl/reviews%f.pkl' % _SAMPLE, 'r'))
  train = load(open('pkl/train%f.pkl' % _SAMPLE, 'r'))
  test = load(open('pkl/test%f.pkl' % _SAMPLE, 'r'))
  overall_avg = float(sum([float(v['vote']) for v in train])) / len(train)
  
  print 'Fitting Model'
  model = TF_Model()
  model.fit(train, reviews)

  print 'Calculating Predictions'
  pred = model.predict(train, reviews)
   
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
  output = open('%s/tf%f.dat' % (_OUTPUT_DIR, _SAMPLE), 'w')
  for p in pred:
    print >> output, overall_avg if isnan(p) else p
  output.close()
