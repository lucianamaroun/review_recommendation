""" BETF Module
    -----------

    Implementation of Tensor Factorization for prediction of helpfulness votes.
    Voter, author and product are the dimensions considered in the model and
    modeled as latent vectors. Interaction between author and product explain
    observed rating as well. Bias of review, rater, author and product is
    distinguished to explain part of the observed variables, vote and rating.

    Usage:
    $ python -m algorithms.betf.prediction [-k <latent_dimensions>]
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

from numpy import nan, isnan, tensordot
from numpy.random import random
from pickle import load

from util.aux import sigmoid, sigmoid_der1
from evaluation.metrics import calculate_rmse, calculate_avg_ndcg


_K = 2
_ITER = 1000    # number of iterations of stochastic gradient descent
_ALPHA = 1      # starting learning rate
_BETA = 0.1     # regularization factor
_TOL = 1e-6 
_PKL_DIR = 'out/pkl'
_VAL_DIR = 'out/val'
_OUTPUT_DIR = 'out/pred/'


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
      print ('Usage: $ python -m algorithms.cf.user_based '
          '[-k <latent_dimensions>] [-l <learning_rate>] [-r <regularization>]'
          '[-e <tolerance>] [-i <iterations>]')
      exit()
    i = i + 2


class BETF_Model(object):
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
    self.voter_bias = None
    self.review_bias = None
    self.vote_avg = None
    self.author_bias = None
    self.product_bias = None
    self.rating_avg = None

  def _initialize_matrices(self, votes, reviews):
    """ Initializes matrices and mappings given votes. Each entity id is mapped
        to an index in a dimension of  the matrix.

        Args:
          votes: list of votes (training set).
        
        Returns:
          None. Instance fields are updated.
    """
    voters = sorted(set([vote['voter'] for vote in votes]))
    authors = sorted(set([vote['author'] for vote in votes]) \
        .union(set([review['author'] for review in reviews.itervalues()])))
    products = sorted(set([reviews[v['review']]['product'] for v in votes]) \
        .union(set([review['product'] for review in reviews.itervalues()])))
    self.voter_map = {u:i for i, u in enumerate(voters)}
    self.author_map = {a:i for i, a in enumerate(authors)}
    self.product_map = {p:i for i, p in enumerate(products)}
    self.V = random((len(voters), _K))
    self.A = random((len(authors), _K))
    self.P = random((len(products), _K))
    self.S = random((_K, _K, _K))

  def _calculate_vote_bias(self, votes):
    self.voter_bias = {}
    voter_count = {}
    self.review_bias = {}
    review_count = {}
    self.overall_mean = 0
    count = 0
    for vote in votes:
      self.overall_mean += vote['vote']
      count += 1
    self.overall_mean /= float(count)
    for vote in votes:
      voter = vote['voter']
      if voter not in self.voter_bias:
        self.voter_bias[voter] = 0
        voter_count[voter] = 0
      self.voter_bias[voter] += (vote['vote'] - self.overall_mean)
      voter_count[voter] += 1
    for voter in self.voter_bias:
      self.voter_bias[voter] /= float(voter_count[voter])
    for vote in votes:
      voter = vote['voter']
      review = vote['review']
      if review not in self.review_bias:
        self.review_bias[review] = 0
        review_count[review] = 0
      self.review_bias[review] += (vote['vote'] - self.overall_mean -
          self.voter_bias[voter]) 
      review_count[review] += 1
    for review in self.review_bias:
      self.review_bias[review] /= float(review_count[review])
#  def _calculate_vote_bias(self, votes):
#    self.voter_bias = {}
#    voter_count = {}
#    self.review_bias = {}
#    review_count = {}
#    self.vote_avg = 0
#    count = 0
#    for vote in votes:
#      voter = vote['voter']
#      review = vote['review']
#      if voter not in self.voter_bias:
#        self.voter_bias[voter] = 0
#        voter_count[voter] = 0
#      self.voter_bias[voter] += vote['vote']
#      voter_count[voter] += 1
#      if review not in self.review_bias:
#        self.review_bias[review] = 0
#        review_count[review] = 0
#      self.review_bias[review] += vote['vote']
#      review_count[review] += 1
#      self.vote_avg += vote['vote']
#      count += 1
#    self.vote_avg /= float(count)
#    for voter in self.voter_bias:
#      self.voter_bias[voter] /= float(voter_count[voter])
#      self.voter_bias[voter] -= self.vote_avg
#    for review in self.review_bias:
#      self.review_bias[review] /= float(review_count[review])
#      self.review_bias[review] -= self.vote_avg 
      
  def _calculate_rating_bias(self, reviews):
    self.author_bias = {}
    author_count = {}
    self.product_bias = {}
    product_count = {}
    self.overall_mean = 0
    count = 0
    for review in reviews:
      self.overall_mean += review['rating']
      count += 1
    self.overall_mean /= float(count)
    for review in reviews:
      author = review['author']
      if author not in self.author_bias:
        self.author_bias[author] = 0
        author_count[author] = 0
      self.author_bias[author] += (review['rating'] - self.overall_mean)
      author_count[author] += 1
    for author in self.author_bias:
      self.author_bias[author] /= float(author_count[author])
    for review in reviews:
      author = review['author']
      product = review['product']
      if product not in self.product_bias:
        self.product_bias[product] = 0
        product_count[product] = 0
      self.product_bias[product] += (review['rating'] - self.overall_mean -
          self.author_bias[author]) 
      product_count[product] += 1
    for product in self.product_bias:
      self.product_bias[product] /= float(product_count[product])
#  def _calculate_review_bias(self, reviews):
#    self.author_bias = {}
#    author_count = {}
#    self.product_bias = {}
#    product_count = {}
#    self.rating_avg = 0
#    count = 0
#    for review in reviews.itervalues():
#      author = review['author']
#      product = review['product']
#      if author not in self.author_bias:
#        self.author_bias[author] = 0
#        author_count[author] = 0
#      self.author_bias[author] += review['rating'] 
#      author_count[author] += 1
#      if product not in self.product_bias:
#        self.product_bias[product] = 0
#        product_count[product] = 0
#      self.product_bias[product] += review['rating']
#      product_count[product] += 1
#      self.rating_avg += review['rating']
#      count += 1
#    self.rating_avg /= float(count)
#    for author in self.author_bias:
#      self.author_bias[author] /= float(author_count[author])
#      self.author_bias[author] -= self.rating_avg 
#    for product in self.product_bias:
#      self.product_bias[product] /= float(product_count[product])
#      self.product_bias[product] -= self.rating_avg 

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
    """ Computes the derivative of the tensor dot relative to 'v' variable.

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
    """ Computes the derivative of the tensor dot relative to 'a' variable.

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
    """ Computes the derivative of the tensor dot relative to 'p' variable.

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
    self._calculate_vote_bias(votes)
    self._calculate_rating_bias(reviews)
    previous = float('inf')
    for it in xrange(_ITER):
      _ALPHA = 1.0 / sqrt(it+1)
      for vote in votes:
        v = self.voter_map[vote['voter']]
        a = self.author_map[vote['author']]
        p = self.product_map[reviews[vote['review']]['product']]
        dot = self.vote_avg + self.voter_bias[vote['voter']] + \
            self.review_bias[vote['review']] + self.tensor_dot(v, a, p)
        error = float(vote['vote']) / 5.0 - sigmoid(dot) # normalized in (0,1)
        der_sig = sigmoid_der1(dot)
        self.voter_bias[vote['voter']] += _ALPHA * 2 * (error - _BETA *
            self.voter_bias[vote['voter']])
        self.review_bias[vote['review']] += _ALPHA * 2 * (error - _BETA *
            self.review_bias[vote['review']])
        self.V[v,:] += _ALPHA * (2 * error * der_sig * \
            self.tensor_dot_der_v(a, p) - _BETA * self.V[v,:])                                          
        self.A[a,:] += _ALPHA * (2 * error * der_sig * \
            self.tensor_dot_der_a(v, p) - _BETA * self.A[a,:])
        self.P[p,:] += _ALPHA * (2 * error * der_sig * \
            self.tensor_dot_der_p(v, a) - _BETA * self.P[p,:])                                          
        self.S += _ALPHA * (2 * error * der_sig * \
            self.tensor_dot_der_s(v, a, p) - _BETA * self.S) 
      for review in reviews.itervalues():
        a = self.author_map[review['author']]
        p = self.product_map[review['product']]
        dot = self.rating_avg + self.author_bias[review['author']] + \
            self.product_bias[review['product']] + self.A[a,:].dot(self.P[p,:]) 
        error = float(review['rating']) / 5.0 - sigmoid(dot) # normalized in (0,1)
        der_sig = sigmoid_der1(dot)
        self.author_bias[review['author']] += _ALPHA * 2 * (error - _BETA *
            self.author_bias[review['author']])
        self.product_bias[review['product']] += _ALPHA * 2 * (error - _BETA *
            self.product_bias[review['product']])
        self.A[a,:] += _ALPHA * (2 * error * der_sig * self.P[p,] - \
            _BETA * self.A[a,:])
        self.P[p,:] += _ALPHA * (2 * error * der_sig * self.A[a,:] - \
            _BETA * self.P[p,:])                                          
      value = 0.0
      for vote in votes:
        v = self.voter_map[vote['voter']]
        a = self.author_map[vote['author']]
        p = self.product_map[reviews[vote['review']]['product']]
        dot = self.vote_avg + self.voter_bias[vote['voter']] + \
            self.review_bias[vote['review']] + self.tensor_dot(v, a, p)
        value += (float(vote['vote']) / 5.0 - sigmoid(dot)) ** 2 # normalized in (0,1)
        value += self.voter_bias[vote['voter']] ** 2 + \
            self.review_bias[vote['review']] ** 2
        for i in xrange(_K):
          value += _BETA * (self.V[v,i] ** 2 + self.A[a,i] ** 2 + \
              self.P[p,i] ** 2)
          for j in xrange(_K):
            for k in xrange(_K):
              value += _BETA * (self.S[i,j,k] ** 2)
      for review in reviews.itervalues():
        a = self.author_map[review['author']]
        p = self.product_map[review['product']]
        dot = self.rating_avg + self.author_bias[review['author']] + \
            self.product_bias[review['product']] + self.A[a,:].dot(self.P[p,:]) 
        value += (float(review['rating']) / 5.0 - sigmoid(dot)) ** 2 # normalized in (0,1)
        value += self.author_bias[review['author']] ** 2 + \
            self.product_bias[review['product']] ** 2
        for k in xrange(_K):
          value += _BETA * (self.V[v,k] ** 2 + self.P[p,k] ** 2)
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
        dot = self.vote_avg + self.voter_bias[vote['voter']] + \
            self.review_bias[vote['review']] + self.tensor_dot(v, a, p)
        pred.append(sigmoid(dot))
      else:
        pred.append(nan)
    return pred


if __name__ == '__main__':
  load_args()

  for i in xrange(1):#NUM_SETS):
    print 'Reading pickles'
    train = load(open('%s/train-%d.pkl' % (_PKL_DIR, i), 'r'))
    val = load(open('%s/validation-%d.pkl' % (_PKL_DIR, i), 'r'))
    test = load(open('%s/test-%d.pkl' % (_PKL_DIR, i), 'r'))
    reviews = load(open('%s/reviews-%d.pkl' % (_PKL_DIR, i), 'r'))
    truth = [v['vote'] for v in train]
    overall_avg = float(sum(truth)) / len(train)
  
    train_reviews_ids = set([vote['review'] for vote in train])
    train_reviews = {r_id:reviews[r_id] for r_id in train_reviews_ids}
   
    print 'Fitting Model'
    model = BETF_Model()
    model.fit(train, train_reviews)

    print 'Calculate Predictions'
    pred = model.predict(train, reviews)
    pred = [p * 5.0 for p in pred] 
    
    print 'TRAINING ERROR'
    print '-- RMSE: %f' % calculate_rmse(pred, truth)
    print '-- nDCG@%d: %f' % (RANK_SIZE, calculate_avg_ndcg(train, reviews, 
        pred, truth, RANK_SIZE))
    
    print 'Outputting Validation Prediction'
    pred = model.predict(test, reviews) 
    output = open('%s/betf-%d.dat' % (_VAL_DIR, i), 'w')
    for p in pred:
      print >> output, overall_avg if isnan(p) else p
    output.close()
    
    print 'Outputting Test Prediction'
    pred = model.predict(test, reviews) 
    output = open('%s/betf-%d.dat' % (_OUTPUT_DIR, i), 'w')
    for p in pred:
      print >> output, overall_avg if isnan(p) else p
    output.close()
