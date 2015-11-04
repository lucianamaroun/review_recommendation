""" MF Module
    ---------

    Implementation of Matrix Factorization for prediction of helpfulness votes.
    Voter and review are the dimensions considered in the model and modeled as
    latent vectors.

    Usage:
      $ python -m algo.cf.mf [-k <k>] [-i <iterations>] [-l <learning_rate>]
        [-r <regularization>] [-e <tolerance>] [-b <bias>]
    where:
    <k> is an integer with the number of latent dimensions,
    <iterations> is an integer with the maximum number of stochastic gradient
      descent iterations,
    <learning_rate> is a float with the update factor of gradient descent,
    <regulatization> is a float with the regularization weight in optimization
      objective,
    <tolerance> is a float with convergence criterion tolerance,
    <bias> is either 'y' or 'n', meaning use and not use bias, respectively.
"""


from math import sqrt
from sys import argv, exit
from time import time
from random import shuffle

from numpy import nan, isnan, copy
from numpy.random import uniform, seed
from pickle import load

from algo.const import NUM_SETS, RANK_SIZE, REP 
from perf.metrics import calculate_rmse, calculate_avg_ndcg


_K = 5
_ITER = 1000      # number of iterations of stochastic gradient descent
_ALPHA = 0.001    # learning rate 
_BETA = 0.01      # regularization factor 
_TOL = 1e-6
_BIAS = False 
_VAL_DIR = 'out/val'
_OUTPUT_DIR = 'out/test'
_PKL_DIR = 'out/pkl'
_CONF_STR = None


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
    elif argv[i] == '-i':
      global _ITER
      _ITER = int(argv[i+1])
    elif argv[i] == '-l':
      global _ALPHA
      _ALPHA = float(argv[i+1])
    elif argv[i] == '-r':
      global _BETA
      _BETA = float(argv[i+1])
    elif argv[i] == '-e':
      global _TOL
      _TOL = float(argv[i+1])
    elif argv[i] == '-b' and argv[i+1] in ['y', 'n']:
      global _BIAS
      _BIAS = True if argv[i+1] == 'y' else False
    else:
      print ('Usage:\n  $ python -m algo.cf.mf [-k <k>] [-i <iterations>] '
          '[-l <learning_rate>] [-r <regularization>] [-e <tolerance>] '
          '[-b <bias>]')
      exit()
    i = i + 2
  global _CONF_STR
  _CONF_STR = 'k:%d,i:%d,l:%f,r:%f,e:%f,b:%s' % (_K, _ITER, _ALPHA, _BETA, _TOL,
      'y' if _BIAS else 'n')


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
    self.R = None # Matrix of author latent arrays (N_a, K)
    self.user_map = None # Map from user ids to matrix indices (lines)
    self.author_map = None # Map from review ids to matrix indices (lines)
    self.author_bias = None
    self.user_bias = None
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
    authors = sorted(set([vote['author'] for vote in votes]))
    self.user_map = {u:i for i, u in enumerate(users)}
    self.author_map = {r:i for i, r in enumerate(authors)}
    seed(int(time() * 1000000) % 1000000)
    self.U = uniform(1e-10, 1e-8, (len(users), _K))
    self.A = uniform(1e-10, 1e-8, (len(authors), _K))
    self.overall_mean = float(sum([v['vote'] for v in votes])) / len(votes)

  def _calculate_bias(self, votes):
    self.user_bias = {}
    user_count = {}
    self.author_bias = {}
    author_count = {}
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
      author = vote['author']
      if author not in self.author_bias:
        self.author_bias[author] = 0
        author_count[author] = 0
      self.author_bias[author] += (vote['vote'] - self.overall_mean -
          self.user_bias[user]) 
      author_count[author] += 1
    for author in self.author_bias:
      self.author_bias[author] /= float(author_count[author])

  def fit(self, votes):
    """ Fits a MF model given training set (votes).

        Args:
          vote: list of votes, represented as dictionaries (training set).
        
        Returns:
          None. Instance fields are updated.
    """
    self._initialize_matrices(votes)
    if _BIAS:
      self._calculate_bias(votes)
    previous = float('inf')
    shuffle(votes)
    for it in xrange(_ITER):
      for vote in votes:
        u = self.user_map[vote['voter']]
        a = self.author_map[vote['author']]
        dot = self.U[u,:].dot(self.A[a,:].T)
        if _BIAS:
          dot += self.overall_mean + self.user_bias[vote['voter']] + \
              self.author_bias[vote['author']]
        error = dot - float(vote['vote'])
        if _BIAS:
          self.user_bias[vote['voter']] -= _ALPHA * (error + _BETA *
              self.user_bias[vote['voter']])
          self.author_bias[vote['author']] -= _ALPHA * (error + _BETA *
              self.author_bias[vote['author']])
        new_u = self.U[u,:] - _ALPHA * (error * self.A[a,:] + \
            _BETA * self.U[u,:])
        new_a = self.A[a,:] - _ALPHA * (error * self.U[u,:] + \
            _BETA * self.A[a,:])
        self.U[u,:] = new_u
        self.A[a,:] = new_a
      value = 0.0
      for vote in votes:
        u = self.user_map[vote['voter']]
        a = self.author_map[vote['author']]
        dot = self.U[u,:].dot(self.A[a,:].T)
        if _BIAS:
          dot += self.overall_mean + self.user_bias[vote['voter']] + \
              self.author_bias[vote['author']]
        value += (vote['vote'] - dot) ** 2
        if _BIAS:
          value += self.user_bias[vote['voter']] ** 2 + \
              self.author_bias[vote['author']] ** 2
        for k in xrange(_K):
          value += _BETA * (self.U[u,k] ** 2 + self.A[a,k] ** 2)
        value /= 2.0
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
    cold_start = 0
    for vote in votes:
      u = self.user_map[vote['voter']] if vote['voter'] in self.user_map else -1
      a = self.author_map[vote['author']] if vote['author'] in self.author_map else -1
      if u != -1 and a != -1:
        dot = self.U[u,:].dot(self.A[a,:].T)
        if _BIAS:
          dot += self.overall_mean + self.user_bias[vote['voter']] + \
              self.author_bias[vote['author']]
        pred.append(dot)
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
    
    for j in xrange(REP):
      print 'Fitting Model'
      model = MF_Model()
      model.fit(train)

      print 'Calculating Predictions'
      pred = model.predict(train)
      print pred[:10] 
      print 'TRAINING ERROR'
      print '-- RMSE: %f' % calculate_rmse(pred, truth)
      print '-- nDCG@%d: %f' % (RANK_SIZE, calculate_avg_ndcg(train, reviews,
          pred, truth, RANK_SIZE))
      
      pred = model.predict(val) 
      print pred[:10] 
      print 'Outputting validation prediction'
      output = open('%s/mf-%s-%d-%d.dat' % (_VAL_DIR, _CONF_STR, i, j), 'w')
      for p in pred:
        print >> output, p
      output.close()
      
      pred = model.predict(test) 
      print pred[:10] 
      print 'Outputting testing prediction'
      output = open('%s/mf-%s-%d-%d.dat' % (_OUTPUT_DIR, _CONF_STR, i, j), 'w')
      for p in pred:
        print >> output, p
      output.close()
