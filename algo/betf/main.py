""" BETF Module
    -----------

    Implementation of Tensor Factorization for prediction of helpfulness votes.
    Voter, author and product are the dimensions considered in the model and
    modeled as latent vectors. Interaction between author and product explain
    observed rating as well. Bias of review, rater, author and product is
    distinguished to explain part of the observed variables, vote and rating.

    Usage:
    $ python -m algo.betf.main [-k <latent_dimensions>]
      [-l <learning_rate>] [-r <regularization>] [-e <tolerance>]
      [-i <iterations>] [-b <bias_type>]
    where
    <latent_dimensions> is an integer with the number of latent dimensions,
    <learning_rate> is a float representing the update rate of gradient descent,
    <regularization> is a float with the weight of regularization in objective
      function,
    <tolerance> is a float with the tolerance for convergence,
    <iterations> is an integer with the maximum number of iterations of gradient
      descent,
    <bias_type> is either 's' static or 'd' for dynamic, being updated in the
      optimization.
"""


from math import sqrt
from sys import argv, exit
from random import shuffle

from numpy import nan, isnan, tensordot, array, mean, vstack
from numpy.random import uniform 
from pickle import load

from algo.const import NUM_SETS, RANK_SIZE, REP
from util.aux import sigmoid, sigmoid_der1
from perf.metrics import calculate_rmse, calculate_avg_ndcg


_K = 2
_ITER = 10      # number of iterations of stochastic gradient descent
_ALPHA = 1      # starting learning rate
_BETA = 0.1     # regularization factor
_TOL = 1e-6
_BIAS = 'd'
_UPDATE_BIAS = True
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
    elif argv[i] == '-b' and argv[i+1] in ['s', 'd']:
      global _BIAS
      _BIAS = argv[i+1]
      _UPDATE_BIAS = True if _BIAS == 'd' else False
    else:
      print ('Usage: $ python -m algo.betf.main '
          '[-k <latent_dimensions>] [-l <learning_rate>] [-r <regularization>] '
          '[-e <tolerance>] [-i <iterations>] [-b <bias_type>]')
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
    self.user_map = None
    self.product_map = None 
    self.voter_bias = None
    self.author_vote_bias = None
    self.product_vote_bias = None
    self.vote_avg = None
    self.author_rating_bias = None
    self.product_rating_bias = None
    self.rating_avg = None
    self.overall_mean = None
    self.mean_voter = None 
    self.mean_author = None 
    self.mean_product = None

  def _initialize_matrices(self, votes, reviews):
    """ Initializes matrices and mappings given votes. Each entity id is mapped
        to an index in a dimension of  the matrix.

        Args:
          votes: list of votes (training set).
        
        Returns:
          None. Instance fields are updated.
    """
    voters = set([vote['voter'] for vote in votes])
    authors = set([vote['author'] for vote in votes]) \
        .union(set([review['author'] for review in reviews.itervalues()]))
    products = set([reviews[v['review']]['product'] for v in votes]) \
        .union(set([review['product'] for review in reviews.itervalues()]))
    self.voter_map = {u:i for i, u in enumerate(voters)}
    self.author_map = {a:i for i, a in enumerate(authors)}
    self.product_map = {p:i for i, p in enumerate(products)}
    self.V = uniform(1e-4, 1e-2, (len(voters), _K))
    self.A = uniform(1e-4, 1e-2, (len(authors), _K))
    self.P = uniform(1e-4, 1e-2, (len(products), _K))
    self.S = uniform(1e-4, 1e-2, (_K, _K, _K))
    self.overall_mean = float(sum([v['vote'] for v in votes])) / len(votes)
  
  def _calculate_vote_bias(self, votes, reviews):
    self.voter_bias = {}
    voter_count = {}
    self.author_vote_bias = {}
    author_count = {}
    self.product_vote_bias = {}
    product_count = {}
    self.vote_avg = 0
    count = 0
    for vote in votes:
      self.vote_avg += vote['vote']
      count += 1
    self.vote_avg /= float(count)
    for vote in votes:
      voter = vote['voter']
      if voter not in self.voter_bias:
        self.voter_bias[voter] = 0
        voter_count[voter] = 0
      self.voter_bias[voter] += (vote['vote'] - self.vote_avg)
      voter_count[voter] += 1
    for voter in self.voter_bias:
      self.voter_bias[voter] /= float(voter_count[voter])
    for vote in votes:
      voter = vote['voter']
      author = vote['author']
      if author not in self.author_vote_bias:
        self.author_vote_bias[author] = 0
        author_count[author] = 0
      self.author_vote_bias[author] += (vote['vote'] - self.vote_avg -
          self.voter_bias[voter]) 
      author_count[author] += 1
    for author in self.author_vote_bias:
      self.author_vote_bias[author] /= float(author_count[author])
    for vote in votes:
      voter = vote['voter']
      author = vote['author']
      product = reviews[vote['review']]['product']
      if product not in self.product_vote_bias:
        self.product_vote_bias[product] = 0
        product_count[product] = 0
      self.product_vote_bias[product] += (vote['vote'] - self.vote_avg -
          self.voter_bias[voter] - self.author_vote_bias[author]) 
      product_count[product] += 1
    for product in self.product_vote_bias:
      self.product_vote_bias[product] /= float(product_count[product])
      
  def _calculate_rating_bias(self, reviews):
    self.author_rating_bias = {}
    author_count = {}
    self.product_rating_bias = {}
    product_count = {}
    self.rating_avg = 0
    count = 0
    for review in reviews.itervalues():
      self.rating_avg += review['rating']
      count += 1
    self.rating_avg /= float(count)
    for review in reviews.itervalues():
      author = review['author']
      if author not in self.author_rating_bias:
        self.author_rating_bias[author] = 0
        author_count[author] = 0
      self.author_rating_bias[author] += (review['rating'] - 
          self.rating_avg)
      author_count[author] += 1
    for author in self.author_rating_bias:
      self.author_rating_bias[author] /= float(author_count[author])
    for review in reviews.itervalues():
      author = review['author']
      product = review['product']
      if product not in self.product_rating_bias:
        self.product_rating_bias[product] = 0
        product_count[product] = 0
      self.product_rating_bias[product] += (review['rating'] -
          self.rating_avg - self.author_rating_bias[author]) 
      product_count[product] += 1
    for product in self.product_rating_bias:
      self.product_rating_bias[product] /= float(product_count[product])
      
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
    dot = array([0.0] * _K)
    for d in xrange(_K):
      for y in xrange(_K):
        for z in xrange(_K):
          dot[d] += self.S[d,y,z] * self.A[a,y] * self.P[p,z]
    return dot
  
  def tensor_dot_der_a(self, v, p):
    """ Computes the derivative of the tensor dot relative to 'a' variable.

        Args:
          v: index of vector in V matrix.
          p: index of vector in P matrix.
    
        Return:
          A k-array with the derivative at each dimension of 'a'.
    """
    dot = array([0.0] * _K)
    for d in xrange(_K):
      for x in xrange(_K):
        for z in xrange(_K):
          dot[d] += self.S[x,d,z] * self.V[v,x] * self.P[p,z]
    return dot
  
  def tensor_dot_der_p(self, v, a):
    """ Computes the derivative of the tensor dot relative to 'p' variable.

        Args:
          v: index of vector in V matrix.
          a: index of vector in A matrix.
    
        Return:
          A k-array with the derivative at each dimension of 'p'.
    """
    dot = array([0.0] * _K)
    for d in xrange(_K):
      for x in xrange(_K):
        for y in xrange(_K):
          dot[d] += self.S[x,y,d] * self.V[v,x] * self.A[a,y]
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
 
  def fit(self, votes, reviews_dict):
    """ Fits a TF model given training set (votes).

        Args:
          vote: list of votes, represented as dictionaries (training set).
          reviews_dict: dictionary of reviews.

        Returns:
          None. Instance fields are updated.
    """
    self._initialize_matrices(votes, reviews_dict)
    self._calculate_vote_bias(votes, reviews_dict)
    self._calculate_rating_bias(reviews_dict)
    previous = float('inf')
    alpha = _ALPHA
    reviews = reviews_dict.values()
    shuffle(votes)
    shuffle(reviews)
    for it in xrange(_ITER):
      alpha = alpha / sqrt(it+1)
      print 'Iteration %d' % it
      for vote in votes:
        voter = vote['voter']
        author = vote['author']
        product = reviews_dict[vote['review']]['product']
        v = self.voter_map[voter]
        a = self.author_map[author]
        p = self.product_map[product]
        dot = self.vote_avg + self.voter_bias[voter] + \
            self.author_vote_bias[author] + self.product_vote_bias[product] + \
            self.tensor_dot(v, a, p)
        error = sigmoid(dot) - vote['vote'] # normalized in (0,1) for sig
        der_sig = sigmoid_der1(dot)
        new_v = self.V[v,:] - alpha * (error * der_sig * \
            self.tensor_dot_der_v(a, p))# + _BETA * self.V[v,:]) 
        new_a = self.A[a,:] - alpha * (error * der_sig * \
            self.tensor_dot_der_a(v, p))# + _BETA * self.A[a,:])
        new_p = self.P[p,:] - alpha * (error * der_sig * \
            self.tensor_dot_der_p(v, a))# + _BETA * self.P[p,:])
        new_s = self.S - alpha * (error * der_sig * \
            self.tensor_dot_der_s(v, a, p))# + _BETA * self.S) 
        self.V[v,:] = new_v
        self.A[a,:] = new_a
        self.P[p,:] = new_p
        self.S = new_s
      for review in reviews:
        author = review['author']
        product = review['product']
        a = self.author_map[author]
        p = self.product_map[product]
        dot = self.rating_avg + self.author_rating_bias[author] + \
            self.product_rating_bias[product] + \
            self.A[a,:].dot(self.P[p,:]) 
        error = sigmoid(dot) - review['rating'] / 5.0 # normalized in (0,1)
        der_sig = sigmoid_der1(dot)
        new_a = self.A[a,:] - alpha * (error * der_sig * self.P[p,:])# + \
            # _BETA * self.A[a,:])
        new_p = self.P[p,:] - alpha * (error * der_sig * self.A[a,:])# + \
            # _BETA * self.P[p,:])
        self.A[a,:] = new_a
        self.P[p,:] = new_p
      self.V -= alpha * _BETA * self.V                                          
      self.A -= alpha * _BETA * self.A
      self.P -= alpha * _BETA * self.P                                          
      self.S -= alpha * _BETA * self.S 
      if _UPDATE_BIAS:
        for voter in self.voter_bias:
          self.voter_bias[voter] -= alpha * _BETA * self.voter_bias[voter]
        for author in self.author_vote_bias:
          self.author_vote_bias[author] -= alpha * _BETA * \
              self.author_vote_bias[author]
        for product in self.product_vote_bias:
          self.product_vote_bias[product] -= alpha * _BETA * \
              self.product_vote_bias[product]
        for author in self.author_rating_bias:
          self.author_rating_bias[author] -= alpha * _BETA * \
              self.author_rating_bias[author]
        for product in self.product_rating_bias:
          self.product_rating_bias[product] -= alpha * _BETA * \
              self.product_rating_bias[product]
      value = 0.0
      for vote in votes:
        voter = vote['voter']
        author = vote['author']
        product = reviews_dict[vote['review']]['product']
        v = self.voter_map[voter]
        a = self.author_map[author]
        p = self.product_map[product]
        dot = self.vote_avg + self.voter_bias[voter] + \
            self.author_vote_bias[author] + self.product_vote_bias[product] + \
            self.tensor_dot(v, a, p)
        value += (vote['vote'] / 5.0 - sigmoid(dot)) ** 2 # normalized in (0,1)
      for review in reviews:
        author = review['author']
        product = review['product']
        a = self.author_map[author]
        p = self.product_map[product]
        dot = self.rating_avg + self.author_rating_bias[author] + \
            self.product_rating_bias[product] + self.A[a,:].dot(self.P[p,:]) 
        value += (review['rating'] / 5.0 - sigmoid(dot)) ** 2 # normalized in (0,1)
      for v in self.voter_map.itervalues():
        for i in xrange(_K):
          value += _BETA * self.V[v,i] ** 2
      for a in self.author_map.itervalues():
        for i in xrange(_K):
          value += _BETA * self.A[a,i] ** 2
      for p in self.product_map.itervalues():
        for i in xrange(_K):
          value += _BETA * self.P[p,i] ** 2
      for i in xrange(_K):
        for j in xrange(_K):
          for k in xrange(_K):
            value += _BETA * self.S[i,j,k] ** 2
      if _UPDATE_BIAS:
        for voter in self.voter_bias:
          value += _BETA * self.voter_bias[voter] ** 2
        for author in self.author_vote_bias:
          value += _BETA * self.author_vote_bias[author] ** 2
        for product in self.product_vote_bias:
          value += _BETA * self.product_vote_bias[product] ** 2
        for author in self.author_rating_bias:
          value += _BETA * self.author_rating_bias[author] ** 2
        for product in self.product_rating_bias:
          value += _BETA * self.product_rating_bias[product] ** 2
      value /= 2.0
      if abs(previous - value) < _TOL:
        print previous
        print value
        print 'Break'
        break
      previous = value
    self.mean_voter = mean([self.V[v,:] for v in self.voter_map.itervalues()],
        axis=0)
    self.mean_author = mean([self.A[a,:] for v in self.author_map.itervalues()],
        axis=0)
    self.mean_product = mean([self.P[p,:] for p in
        self.product_map.itervalues()], axis=0)
    self.V = vstack((self.V, self.mean_voter.reshape(1, _K)))
    self.A = vstack((self.A, self.mean_author.reshape(1, _K)))
    self.P = vstack((self.P, self.mean_product.reshape(1, _K)))

  def predict(self, votes, reviews):
    """ Predicts a set of vote examples using previous fitted model.

        Args:
          votes: list of dictionaries, representing votes, to predict
        helpfulness vote value.

        Returns:
          A list of floats with predicted vote values.
    """
    pred = []
    cold_start = 0
    for vote in votes:
      voter = vote['voter']
      author = vote['author']
      product = reviews[vote['review']]['product']
      v = self.voter_map[voter] if voter in self.voter_map else -1
          # last position contains mean latent vector1
      a = self.author_map[author] if author in self.author_map else -1
      p = self.product_map[product] if product in self.product_map else -1
      if v != -1 and a != -1 and p != -1:
        dot = self.vote_avg + self.voter_bias[voter] + \
            self.author_vote_bias[author] + self.product_vote_bias[product] + \
            self.tensor_dot(v, a, p)
        pred.append(5.0 * sigmoid(dot))
      else:
        pred.append(self.overall_mean)
        cold_start += 1
     # if v == -1 and a == -1 and p == -1:
     #   pred.append(self.overall_mean)
     # else:
     #   dot = self.vote_avg
     #   if v != -1:
     #     dot += self.voter_bias[voter]
     #   if a != -1:
     #     dot += self.author_vote_bias[author]
     #   if p != -1:
     #     dot += self.product_vote_bias[product]
     #   dot += self.tensor_dot(v, a, p)
     #   pred.append(5.0 * sigmoid(dot))
     # if v == -1 or a == -1 or p == -1:
     #   cold_start += 1
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
      model = BETF_Model()
      model.fit(train, train_reviews)

      print 'Calculating Predictions'
      
      pred = model.predict(train, reviews)
      print 'TRAINING ERROR'
      print '-- RMSE: %f' % calculate_rmse(pred, truth)
      print '-- nDCG@%d: %f' % (RANK_SIZE, calculate_avg_ndcg(train, reviews, 
          pred, truth, RANK_SIZE))

      pred = model.predict(val, reviews) 
      print 'Outputting Validation Prediction'
      output = open('%s/betf-k:%d,l:%f,r:%f,e:%f,i:%d,b:%s-%d-%d.dat' % (_VAL_DIR,
          _K, _ALPHA, _BETA, _TOL, _ITER, _BIAS, i, j), 'w')
      for p in pred:
        print >> output, p
      output.close()
      truth = [v['vote'] for v in val]
      print '-- RMSE: %f' % calculate_rmse(pred, truth)
      print '-- nDCG@%d: %f' % (RANK_SIZE, calculate_avg_ndcg(val, reviews, 
          pred, truth, RANK_SIZE))
      
      pred = model.predict(test, reviews) 
      print 'Outputting Test Prediction'
      output = open('%s/betf-k:%d,l:%f,r:%f,e:%f,i:%d,b:%s-%d-%d.dat' % \
          (_OUTPUT_DIR, _K, _ALPHA, _BETA, _TOL, _ITER, _BIAS, i, j), 'w')
      for p in pred:
        print >> output, p
      output.close()
      truth = [v['vote'] for v in test]
      print '-- RMSE: %f' % calculate_rmse(pred, truth)
      print '-- nDCG@%d: %f' % (RANK_SIZE, calculate_avg_ndcg(test, reviews, 
          pred, truth, RANK_SIZE))
