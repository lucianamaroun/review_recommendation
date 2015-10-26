""" TF Module
    ---------

    Implementation of Tensor Factorization for prediction of helpfulness votes.
    Voter, author and product are the dimensions considered in the model and
    modeled as latent vectors.

    Usage:
    $ python -m algo.recsys.tf [-k <latent_dimensions>]
      [-l <learning_rate>] [-r <regularization>] [-e <tolerance>]
      [-i <iterations>]
    where
    <latent_dimensions> is an integer with the number of latent dimensions,
    <learning_rate> is a float representing the update rate of gradient descent,
    <regularization> is a float with the weight of regularization in objective
      function,
    <tolerance> is a float with the tolerance for convergence,
    <iterations> is an integer with the maximum number of iterations of gradient
      descent.
"""


from math import sqrt
from sys import argv, exit
from random import shuffle

from numpy import array, nan, isnan, tensordot
from numpy.random import uniform 
from pickle import load

from algo.const import NUM_SETS, RANK_SIZE, REP
from util.aux import sigmoid, sigmoid_der1
from perf.metrics import calculate_rmse, calculate_avg_ndcg


_K = 2
_ITER = 1000    # number of iterations of stochastic gradient descent
_ALPHA = 0.01      # starting learning rate
_BETA = 0.1     # regularization factor
_TOL = 1e-6
_PKL_DIR = 'out/pkl'
_VAL_DIR = 'out/val'
_OUTPUT_DIR = 'out/pred'


def load_args():
  """ Loads arguments.

      Args:
        None.

      Returns:
        None. Global variables are updated.      
  """
  i = 1
  while i < len(argv): 
    if argv[i] == '-k':
      global _K
      _K = int(argv[i+1])
    elif argv[i] == '-l':
      global _ALPHA
      _ALPHA = float(argv[i+1])
    elif argv[i] == '-r':
      global _BETA
      _BETA = float(argv[i+1])
    elif argv[i] == '-e':
      global _TOL
      _TOL = float(argv[i+1])
    elif argv[i] == '-i':
      global _ITER
      _ITER = int(argv[i+1])
    else:
      print ('Usage: $ python -m algo.recys.tf '
          '[-k <latent_dimensions>] [-l <learning_rate>] [-r <regularization>] '
          '[-e <tolerance>] [-i <iterations>]')
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
    self.V = None # Matrix of user (voter) latent arrays (N_v, _K)
    self.A = None # Matrix of user (author) latent arrays (N_a, _K)
    self.P = None # Matrix of product latent arrays (N_p, _K)
    self.S = None # Central tensor
    self.voter_map = None 
    self.author_map = None 
    self.product_map = None 
    self.overall_mean = None

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
    self.V = uniform(0, 1, (len(voters), _K))
    self.A = uniform(0, 1, (len(authors), _K))
    self.P = uniform(0, 1, (len(products), _K))
    self.S = uniform(0, 1, (_K, _K, _K))
    self.overall_mean = float(sum([v['vote'] for v in votes])) / len(votes)

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
    for x in xrange(_K):
      for y in xrange(_K):
        for z in xrange(_K):
          dot += self.S[x,y,z] * self.V[v,x] * self.A[a,y] * self.P[p,z]
    return dot
  
  def tensor_dot_der_v(self, a, p):
    """ Computes the derivative of the tensor dot relative to variable 'v'.

        Args:
          a: index of vector in A matrix.
          p: index of vector in P matrix.
    
        Return:
          A k-array with the derivative at each dimension of 'v'.
    """
    dot = 0.0
    for y in xrange(_K):
      for z in xrange(_K):
        dot += self.S[:,y,z] * self.A[a,y] * self.P[p,z]
    return dot
  
  def tensor_dot_der_a(self, v, p):
    """ Computes the derivative of the tensor dot relative to variable 'a'.

        Args:
          v: index of vector in V matrix.
          p: index of vector in P matrix.
    
        Return:
          A k-array with the derivative at each dimension of 'a'.
    """
    dot = 0.0
    for x in xrange(_K):
      for z in xrange(_K):
        dot += self.S[x,:,z] * self.V[v,x] * self.P[p,z]
    return dot
  
  def tensor_dot_der_p(self, v, a):
    """ Computes the derivative of the tensor dot relative to variable 'p'.

        Args:
          v: index of vector in V matrix.
          a: index of vector in A matrix.
    
        Return:
          A k-array with the derivative at each dimension of 'p'.
    """
    dot = 0
    for x in xrange(_K):
      for y in xrange(_K):
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
    dot = array([[[0.0] * _K] * _K] * _K)
    for x in xrange(_K):
      for y in xrange(_K):
        for z in xrange(_K):
          dot[x,y,z] = self.V[v,x] * self.A[a,y] * self.P[p,z]
    return dot
 
  def fit(self, votes, reviews):
    """ Fits a TF model given training set (votes).

        Args:
          vote: list of votes, represented as dictionaries (training set).
        
        Returns:
          None. Instance fields are updated.
    """
    self._initialize_matrices(votes, reviews)
    previous = float('inf')
    shuffle(votes)
    for it in xrange(_ITER):
      print 'Iteration %d' % (it + 1)
      count = 0
      for vote in votes:
        v = self.voter_map[vote['voter']]
        a = self.author_map[vote['author']]
        p = self.product_map[reviews[vote['review']]['product']]
        dot = self.tensor_dot(v, a, p)
        error = dot - float(vote['vote'])
        if isnan(error) or isnan(dot):
          print self.V[v,:]
          print self.A[a,:]
          print self.P[p,:]
          print self.S
          import sys
          sys.exit()
        der_v = self.tensor_dot_der_v(a, p)
        der_a = self.tensor_dot_der_a(v, p)
        der_p = self.tensor_dot_der_p(v, a)
        der_s = self.tensor_dot_der_s(v, a, p)
       # print 'error: %f' % error
       # print 'derivative v: ',
       # print self.tensor_dot_der_v(a, p)
       # print 'error term: ',
       # print error * self.tensor_dot_der_v(a, p)
       # print 'regularization: ',
       # print _BETA * self.V[v,:]
       # print 'update: ',
       # print - _ALPHA * 2 * (error * \
       #     self.tensor_dot_der_v(a, p) + _BETA * self.V[v,:])
       # print 'old v: ',
       # print self.V[v,:]
        new_v = self.V[v,:] - _ALPHA * (error * der_v + _BETA * self.V[v,:])
       # print 'new v: ',
       # print self.V[v,:]
        new_a = self.A[a,:] - _ALPHA * (error * der_a + _BETA * self.A[a,:])
        new_p = self.P[p,:] - _ALPHA * (error * der_p + _BETA * self.P[p,:])
       # print 'derivative s:',
       # print self.tensor_dot_der_s(v, a, p)
       # print 'old s: ',
       # print self.S
        new_s = self.S - _ALPHA * (error * der_s + _BETA * self.S) 
        self.V[v,:] = new_v
        self.A[a,:] = new_a
        self.P[p,:] = new_p
        self.S = new_s
       # new_dot = self.tensor_dot(v, a, p)
       # print 'new s: ',
       # print self.S
       # print 'old dot: ',
       # print dot
       # print 'new dot: ',
       # print new_dot
        count += 1
      value = 0.0
      for vote in votes:
        v = self.voter_map[vote['voter']]
        a = self.author_map[vote['author']]
        p = self.product_map[reviews[vote['review']]['product']]
        dot = self.tensor_dot(v, a, p)
        value += (float(vote['vote']) - dot) ** 2
        for i in xrange(_K):
          value += _BETA * (self.V[v,i] ** 2 + self.A[a,i] ** 2 + \
              self.P[p,i] ** 2)
          for j in xrange(_K):
            for k in xrange(_K):
              value += _BETA * (self.S[i,j,k] ** 2)
        value /= 2.0
      if abs(previous - value) < _TOL:
        print 'Break'
        break
      previous = value

  def predict(self, votes, reviews):
    """ Predicts a set of vote examples using previous fitted model.

        Args:
          votes: list of dictionaries, representing votes, to predict
        helpfulness vote value.
          reviews: dictionary of reviews.

        Returns:
          A list of floats with predicted vote values.
    """
    pred = []
    cold_start = 0
    for vote in votes:
      v = self.voter_map[vote['voter']] if vote['voter'] in self.voter_map \
          else -1
      a = self.author_map[vote['author']] if vote['author'] in \
          self.author_map else -1
      p = self.product_map[reviews[vote['review']]['product']] if \
          reviews[vote['review']]['product'] in self.product_map else -1
      if v != -1 and a != -1 and p != -1:
        pred.append(self.tensor_dot(v, a, p))
      else:
        pred.append(self.overall_mean)
        cold_start += 1
    print 'Cold-start ratio: %f' % (float(cold_start) / len(votes))
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
  
    train_reviews_ids = set([vote['review'] for vote in train])
    train_reviews = {r_id:reviews[r_id] for r_id in train_reviews_ids}
  
    for j in xrange(REP):
      print 'Fitting Model'
      model = TF_Model()
      model.fit(train, train_reviews)

      print 'Calculating Predictions'
      
      pred = model.predict(train, reviews)
      print 'TRAINING ERROR'
      print pred[:10]
      print '-- RMSE: %f' % calculate_rmse(pred, truth)
      print '-- nDCG@%d: %f' % (RANK_SIZE, calculate_avg_ndcg(train, reviews, 
          pred, truth, RANK_SIZE))

      pred = model.predict(val, reviews) 
      print 'Outputting Validation Prediction'
      print pred[:10]
      output = open('%s/tf-k:%d,l:%f,r:%f,e:%f,i:%d-%d-%d.dat' % (_VAL_DIR,
          _K, _ALPHA, _BETA, _TOL, _ITER, i, j), 'w')
      for p in pred:
        print >> output, p
      output.close()
      truth = [v['vote'] for v in val]
      print '-- RMSE: %f' % calculate_rmse(pred, truth)
      print '-- nDCG@%d: %f' % (RANK_SIZE, calculate_avg_ndcg(val, reviews, 
          pred, truth, RANK_SIZE))
      
      pred = model.predict(test, reviews) 
      print 'Outputting Test Prediction'
      print pred[:10]
      output = open('%s/tf-k:%d,l:%f,r:%f,e:%f,i:%d-%d-%d.dat' % \
          (_OUTPUT_DIR, _K, _ALPHA, _BETA, _TOL, _ITER, i, j), 'w')
      for p in pred:
        print >> output, p
      output.close()
      truth = [v['vote'] for v in test]
      print '-- RMSE: %f' % calculate_rmse(pred, truth)
      print '-- nDCG@%d: %f' % (RANK_SIZE, calculate_avg_ndcg(test, reviews, 
          pred, truth, RANK_SIZE))
