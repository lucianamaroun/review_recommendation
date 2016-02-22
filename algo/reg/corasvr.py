""" MF Module
    ---------

    Usage:
      $ python -m algo.reg.corasvr [-i <iterations>] [-l <learning_rate>] 
        [-r <regularization>] [-e <eps_tolerance>] [-g <gamma>] [-p <pow>]
        [-f <feature_set>] [-b <bias>]
    where:
    <iterations> is the number of iterations, one per each update,
    <learning_rate> is a float with the update factor of gradient descent,
    <regulatization> is a float with the regularization weight in optimization
      objective,
    <eps_tolerance> is a float with the tolerance for errors, 
    <gamma> is the trade-off between regression and ranking, the higher the more
      regression is emphasized,
    <pow> is the power to which learning rate is updated, as 
      <learning_rate> / t^<pow>, such that t is the number of updates made,
    <feature_set> is the set of features to use, defined in algo/const.py,
    <bias> is either 'y' or 'n', meaning use and not use bias, respectively.
"""


from math import sqrt, log
from sys import argv, exit
from time import time
from random import shuffle
from pickle import load

from numpy import nan, isnan, copy, zeros, array, ones, shape, column_stack
from numpy.random import uniform, seed, random, shuffle, random_integers, choice

from algo.const import NUM_SETS, RANK_SIZE, REP, REVIEW_FEATS, \
    AUTHOR_FEATS, VOTER_FEATS, SIM_FEATS, CONN_FEATS
from perf.metrics import calculate_rmse, calculate_avg_ndcg
from util.avg_model import compute_avg_user, compute_avg_model 
from util.aux import sigmoid, sigmoid_der1
from util.bias import BiasModel
from util.scaling import fit_scaler, fit_scaler_by_query, scale_features


_ITER = 1000000 # number of iterations of stochastic gradient descent
_ALPHA = 0.01   # learning rate 
_BETA = 0.0001  # regularization factor 
_EPS = 1e-6
_BIAS = False 
_GAMMA = 0.5
_EPS = 0.1
_CHUNK = 1
_POW = 0.25
_FEAT_TYPE = 'cap'

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
    if argv[i] == '-i':
      global _ITER
      _ITER = int(argv[i+1])
    elif argv[i] == '-l':
      global _ALPHA
      _ALPHA = float(argv[i+1])
    elif argv[i] == '-r':
      global _BETA
      _BETA = float(argv[i+1])
    elif argv[i] == '-e':
      global _EPS
      _EPS = float(argv[i+1])
    elif argv[i] == '-b' and argv[i+1] in ['y', 'n']:
      global _BIAS
      _BIAS = True if argv[i+1] == 'y' else False
    elif argv[i] == '-g':
      global _GAMMA
      _GAMMA = float(argv[i+1])
    elif argv[i] == '-p':
      global _POW
      _POW = float(argv[i+1])
    elif argv[i] == '-f' and argv[i+1] in REVIEW_FEATS:
      global _FEAT_TYPE
      _FEAT_TYPE = int(argv[i+1])
    else:
      print ('Usage:\n  $ python -m algo.cf.mf [-i <iterations>] '
          '[-l <learning_rate>] [-r <regularization>] [-e <eps_tolerance>] '
          '[-g <gamma>] [-p <pow>] [-f <feature_set>] [-b <bias>]')
      exit()
    i = i + 2
  global _CONF_STR
  _CONF_STR = 'i:%d,l:%f,r:%f,e:%f,g:%f,p:%f,f:%s,b:%s' % (_ITER, _ALPHA, _BETA, 
      _EPS, _GAMMA, _POW, _FEAT_TYPE, 'y' if _BIAS else 'n')


class CoRaSVR_Model(object):
  """ Class implementing Combined Ranking and SVR - CoRaSVR. """

  def __init__(self):
    """ Discriminates existing attribute, initilizing to None.

        Args:
          None.

        Returns:
          None.
    """
    self.w = None # vector of weights    

  def _initialize_coef(self, X):
    """ Initializes coefficients and adds constant to matrix of features.

        Args:
          X: numpy array with feature arrays in rows (training set).
        
        Returns:
          Matrix of features as an array and with additional constant dimension. 
    """
    m, n = shape(X)
    print '# features: %d' % n
    X = column_stack((ones((m, 1)), array(X)))
    self.w = zeros((n + 1,))#random((n + 1,))
    return X

  def fit(self, X, y, qid):
    """ Fits a model given training set (votes).

        Args:
          X: numpy array with feature arrays in rows (training set).
          y: list of responses, in the same order of X.
          qid: list of query ids (associated to each reader-product pair), in 
            the same order of X.
        
        Returns:
          None. Instance fields are updated.
    """
    X = self._initialize_coef(X)
    m, n = shape(X)
    X_qid = {}
    y_qid = {}
    pairs = []
    for i in xrange(m):
      if qid[i] not in X_qid:
        X_qid[qid[i]] = []
        y_qid[qid[i]] = []
      X_qid[qid[i]].append(i)
      y_qid[qid[i]].append(i)
    count = 0
    for qid in X_qid:
      max_one = max([y[i] for i in X_qid[qid]])
      rest = [y[i] for i in X_qid[qid] if y[i] < max_one]
      max_two = max(rest) if len(rest) > 0 else max_one
      for i in X_qid[qid]:
        for j in X_qid[qid]:
          if i < j and (y[i] >= max_two or y[j] >= max_two):
            pairs.append((i, j))
    p = len(pairs)
    shuffle(pairs)
    t = 1.0
    check = _ITER / 10
    for it in xrange(_ITER):
      alpha = _ALPHA / pow(t, _POW)
      grad = zeros(self.w.shape)
      p_index = random_integers(0, p-1)
      i, j = pairs[p_index]
      i = random_integers(0, m-1)
      true_i = y[i]
      true_j = y[j]
      hyp_i = self.w.dot(X[i])
      if true_i - hyp_i > _EPS and true_i != 0:
        grad += - (_GAMMA) * X[i,:] * true_i 
      elif true_i - hyp_i < - _EPS and true_i != 0:
        grad += (_GAMMA) * X[i,:] * true_i
      hyp_j = self.w.dot(X[j])
      if true_j - hyp_j > _EPS and true_j != 0:
        grad += - (_GAMMA) * X[j,:] * true_j
      elif true_j - hyp_j < - _EPS and true_j != 0:
        grad += (_GAMMA) * X[j,:] * true_j
      x = X[i] - X[j]
      dot = self.w.dot(x)
      delta = 2**y[i] - 2**y[j]
      true = (31 + delta) / 62
      if true_i != 0 and true_j != 0:
        grad += (1 - _GAMMA) * (sigmoid(dot) - true) * x * true 
      self.w[0] -= alpha * grad[0]
      self.w[1:] = max(0.0, 1.0 - alpha * _BETA) * self.w[1:] - alpha * \
            grad[1:]
      t += 1.0
      if (it + 1) % check == 0:
        shuffle(pairs)
        value = 0.0
        for i in xrange(m):
          hyp = self.w.dot(X[i,:])
          value += max(abs(hyp - float(y[i])) - _EPS, 0) * y[i] 
        value += 0.5 * _BETA * (self.w[1:].dot(self.w[1:])) # L2 norm
        value /= m 
        print 'Obj. Fun. (Reg) on iteration %d: %f' % (it, value)
  
  def predict(self, X):
    """ Predicts a set of vote examples using previous fitted model.

        Args:
          X: numpy array with a vector of features per row.

        Returns:
          A list of floats with predicted vote values.
    """
    pred = []
    m = shape(X)[0]
    n = shape(X)[1]
    X = column_stack((ones((m, 1)), X))
    for i in xrange(m):
      pred.append(self.w.T.dot(X[i,:]))
    return pred

def generate_input(reviews, users, sim, conn, votes, avg_user, avg_sim, avg_conn):
  """ Generates input for the regression problem by turning all features
      associated to each entity into a vector. 

      Args:
        reviews: dictionary of reviews.
        users: dictionary of users.
        sim: dictionary of author-voter similary, indexed by the pair.
        conn: dictionary of author-voter connection strength, indexed by the
          pair.
        votes: list of votes to extract features from.

      Returns:
        A triple with an list of features' lists, a list of true votes and a
      list with query ids. 
  """
  X = []
  y = []
  qid = []
  for vote in votes:
    example = []
    review = reviews[vote['review']]
    for feature in REVIEW_FEATS[_FEAT_TYPE]:
      example.append(review[feature])
    author = users[vote['author']] if vote['author'] in users else avg_user
    for feature in AUTHOR_FEATS[_FEAT_TYPE]:
      if isnan(author[feature]):
        example.append(avg_user[feature]) 
      else:
        example.append(author[feature])
    voter = users[vote['voter']] if vote['voter'] in users else avg_user
    for feature in VOTER_FEATS[_FEAT_TYPE]:
      if isnan(voter[feature]):
        example.append(avg_user[feature]) 
      else:
        example.append(voter[feature]) 
    av = (author['id'], voter['id'])
    u_sim = sim[av] if av in sim else avg_sim
    for feature in SIM_FEATS[_FEAT_TYPE]:
      if isnan(u_sim[feature]):
        example.append(avg_sim[feature]) 
      else:
        example.append(u_sim[feature]) 
    u_conn = conn[av] if av in conn else avg_conn
    for feature in CONN_FEATS[_FEAT_TYPE]:
      if isnan(u_conn[feature]):
        example.append(avg_conn[feature]) 
      else:
        example.append(u_conn[feature]) 
    X.append(example)
    y.append(vote['vote'])
    qid.append(vote['voter'])
  return X, y, qid


def main():
  """ Predicts helpfulness votes using MF.

      Args:
        None.

      Returns:
        None. Results are printed to files.
  """
  load_args()
  
  for i in xrange(NUM_SETS):
    t = time()
    print 'Reading pickles'
    train = load(open('%s/train-%d.pkl' % (_PKL_DIR, i), 'r'))
    val = load(open('%s/validation-%d.pkl' % (_PKL_DIR, i), 'r'))
    test = load(open('%s/test-%d.pkl' % (_PKL_DIR, i), 'r'))
    reviews = load(open('%s/new-reviews-%d.pkl' % (_PKL_DIR, i), 'r'))
    users = load(open('%s/users-%d.pkl' % (_PKL_DIR, i), 'r'))
    sim = load(open('%s/new-sim-%d.pkl' % (_PKL_DIR, i), 'r'))
    conn = load(open('%s/new-conn-%d.pkl' % (_PKL_DIR, i), 'r'))
    truth = [v['vote'] for v in train]
    
    if _BIAS:
      bias = BiasModel()
      train = bias.fit_transform(train, reviews)

    avg_user = compute_avg_user(users)
    avg_sim = compute_avg_model(sim)
    avg_conn = compute_avg_model(conn)
    X_train, y_train, qid_train = generate_input(reviews, users, sim, conn,
        train, avg_user, avg_sim, avg_conn)
    X_val, _, qid_val = generate_input(reviews, users, sim, conn, val, avg_user,
        avg_sim, avg_conn)
    X_test, _, qid_test = generate_input(reviews, users, sim, conn, test, 
        avg_user, avg_sim, avg_conn)
    
    scaler = fit_scaler('minmax', X_train)
    X_train = scale_features(scaler, X_train)
    X_val = scale_features(scaler, X_val)
    X_test = scale_features(scaler, X_test)
    print 'Formatting input time: %f' % (time() - t)

    for j in xrange(REP):
      print 'Fitting Model'
      t = time()
      model = LR_Model()
      model.fit(X_train, y_train, qid_train)
      print 'Learning time: %f' % (time() - t)
      print 'Coefficients:'
      print model.w

      print 'Calculating Predictions'
      pred = model.predict(X_train)
      if _BIAS:
        bias.add_bias(train, reviews, pred)

      print 'TRAINING ERROR'
      print '-- RMSE: %f' % calculate_rmse(pred, truth)
      print '-- nDCG@%d: %f' % (RANK_SIZE, calculate_avg_ndcg(train, reviews,
          pred, truth, RANK_SIZE))
      
      pred = model.predict(X_val) 
      if _BIAS:
        bias.add_bias(val, reviews, pred)
      print 'Outputting validation prediction'
      output = open('%s/corasvr-%s-%d-%d.dat' % (_VAL_DIR, _CONF_STR, i, j), 'w')
      for p in pred:
        print >> output, p
      output.close()
     
      t = time()
      pred = model.predict(X_test)
      if _BIAS:
        bias.add_bias(test, reviews, pred)
      print 'Prediction time: %f' % (time() - t)
      print 'Outputting testing prediction'
      output = open('%s/corasvr-%s-%d-%d.dat' % (_OUTPUT_DIR, _CONF_STR, i, j), 
          'w')
      for p in pred:
        print >> output, p
      output.close()


if __name__ == '__main__':
  main()
