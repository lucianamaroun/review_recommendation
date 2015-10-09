""" Prediction using Regression Methods
    -----------------------------------

    Use of regression methods for predicting relevance of reviews for users.

    Usage:
      $ python -m algorithms.regression.gbrt [-r <learning_rate>]
        [-t <num_trees>] [-d <max_depth>] [-l <loss_func>] [-p <subsample>]
        [-m <max_feat>] [-s <scale>] [-f <feature_set>] [-b <bias>]
    where
    <learning_rate> is a float with the gradient update,
    <num_trees> is an integer with the number of decision trees used, 
    <max_depth> is an integer with the maximum depth of trees,
    <loss_func> is the loss function to adjust the data in the set ['ls', 'lad',
      'huber', 'quantile'],
    <subsample> is a float with sample ratio when building each tree, allowing
      randomization,
    <max_feat> is an integer with the number of features used in each tree
      building, allowing randomization,
    <scale> is the type of feature scaling in the set ['all', 'up'] (the first
      means one scaler for all and the second, one scaler for each user and 
      product pair for user and product dependent features),
    <feature_set> is in the set ['www', 'cap', 'all'], and 
    <bias> is either 'y' or 'n'.
"""


from pickle import load
from sys import argv, exit
from time import time

from numpy import nan, isnan
from numpy.random import seed
from sklearn.ensemble import GradientBoostingRegressor

from algorithms.const import NUM_SETS, RANK_SIZE, REP
from evaluation.metrics import calculate_rmse, calculate_avg_ndcg
from util.avg_model import compute_avg_user, compute_avg_model 
from util.scaling import fit_scaler, fit_scaler_by_query, scale_features


_OUTPUT_DIR = 'out/pred'
_VAL_DIR = 'out/val'
_PKL_DIR = 'out/pkl'
_ALPHA = 0.1
_T = 100 
_MAX_D = 3
_LOSS = 'ls' 
_SUBSAMPLE = 1.0
_MAX_F = None
_SCALE = 'all'
_FEAT_TYPE = 'all'
_BIAS = False

_REVIEW_FEATS = {
  'www': ['num_tokens', 'num_sents', 'uni_ratio', 'avg_sent',
    'cap_sent', 'noun_ratio', 'adj_ratio', 'comp_ratio', 'verb_ratio',
    'adv_ratio', 'fw_ratio', 'sym_ratio', 'num_ratio', 'punct_ratio', 'kl',
    'pos_ratio', 'neg_ratio'],
  'cap': ['num_tokens', 'num_sents', 'uni_ratio', 'avg_sent',
    'cap_sent', 'noun_ratio', 'adj_ratio', 'comp_ratio', 'verb_ratio',
    'adv_ratio', 'fw_ratio', 'sym_ratio', 'num_ratio', 'punct_ratio', 'kl',
    'pos_ratio', 'neg_ratio'],
  'all': ['num_tokens', 'num_sents', 'uni_ratio', 'avg_sent',
    'cap_sent', 'noun_ratio', 'adj_ratio', 'comp_ratio', 'verb_ratio',
    'adv_ratio', 'fw_ratio', 'sym_ratio', 'num_ratio', 'punct_ratio', 'kl',
    'pos_ratio', 'neg_ratio'],
}
_AUTHOR_FEATS = {
  'www': ['num_reviews', 'avg_rating', 'num_trustors', 'num_trustees',
    'pagerank'],
  'cap': ['num_reviews', 'avg_rating', 'num_trustors', 'num_trustees',
    'pagerank'],
  'all': ['num_reviews', 'avg_rating', 'num_trustors', 'num_trustees',
    'pagerank', 'num_votes_rec', 'num_votes_giv', 'sd_rating', 'sd_help_rec',
    'sd_help_giv', 'avg_rel_rating', 'avg_help_rec', 'avg_help_giv',
    'avg_rel_help_giv', 'avg_rating_sim', 'avg_help_giv_sim',
    'avg_rating_dir_net', 'avg_help_giv_tru_net']
}
_VOTER_FEATS = {
  'www': [],
  'cap': ['num_trustors', 'num_trustees', 'pagerank', 'avg_rating',
    'avg_rating_dir_net', 'avg_rating_sim', 'avg_help_giv', 'avg_help_giv_sim', 
    'avg_help_giv_tru_net'],
  'all': ['num_reviews', 'avg_rating', 'num_trustors', 'num_trustees',
    'pagerank', 'num_votes_rec', 'num_votes_giv', 'sd_rating', 'sd_help_rec',
    'sd_help_giv', 'avg_rel_rating', 'avg_help_rec', 'avg_help_giv',
    'avg_rel_help_giv', 'avg_rating_sim', 'avg_help_giv_sim',
    'avg_rating_dir_net', 'avg_help_giv_tru_net']
}
_SIM_FEATS = {
  'www': [],
  'cap': ['common_rated', 'jacc_rated', 'cos_ratings', 'pear_ratings',
    'diff_avg_ratings', 'diff_max_ratings', 'diff_min_ratings'],
  'all': ['common_rated', 'jacc_rated', 'cos_ratings', 'pear_ratings',
    'diff_avg_ratings', 'diff_max_ratings', 'diff_min_ratings']
}
_CONN_FEATS = {
  'www': [],
  'cap': ['jacc_trustees', 'jacc_trustors', 'adamic_adar_trustees',
    'adamic_adar_trustors', 'katz'],
  'all': ['jacc_trustees', 'jacc_trustors', 'adamic_adar_trustees',
    'adamic_adar_trustors', 'katz'],
}


def load_args():
  """ Loads arguments.

      Args:
        None.

      Returns:
        A float with the sample size, an integer with dimension K.
  """
  i = 1
  while i < len(argv): 
    if argv[i] == '-r':
      global _ALPHA
      _ALPHA = float(argv[i+1])
    elif argv[i] == '-t':
      global _T
      _T = int(argv[i+1])
    elif argv[i] == '-d':
      global _MAX_D
      _MAX_D = int(argv[i+1])
    elif argv[i] == '-l' and argv[i+1] in ['ls', 'lad', 'huber', 'quantile']:
      global _LOSS 
      _LOSS = argv[i+1]
    elif argv[i] == '-p':
      global _SUBSAMPLE
      _SUBSAMPLE = float(argv[i+1])
    elif argv[i] == '-m':
      global _MAX_F
      _MAX_F = int(argv[i+1])
    elif argv[i] == '-s' and argv[i+1] in ['all', 'up']:
      global _SCALE 
      _SCALE = argv[i+1]
    elif argv[i] == '-f' and argv[i+1] in ['www', 'cap', 'all']:
      global _FEAT_TYPE
      _FEAT_TYPE = argv[i+1]
    elif argv[i] == '-b' and argv[i+1] in ['y', 'n']:
      global _BIAS
      _BIAS = True if argv[i+1] == 'y' else False
    else:
      print argv[i]
      print argv[i+1]
      print ('Usage $ python -m algorithms.regression.gbrt [-r <learning_rate>]'
          '[-t <num_trees>] [-d <max_depth>] [-l <loss_func>] [-p <subsample>]'
          '[-m <max_feat>] [-s <scale>] [-f <feature_set>] [-b <bias>]')
      exit()
    i = i + 2


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
    for feature in _REVIEW_FEATS[_FEAT_TYPE]:
      example.append(review[feature])
    author = users[vote['author']] if vote['author'] in users else avg_user
    for feature in _AUTHOR_FEATS[_FEAT_TYPE]:
      if isnan(author[feature]):
        example.append(avg_user[feature]) 
      else:
        example.append(author[feature])
    voter = users[vote['voter']] if vote['voter'] in users else avg_user
    for feature in _VOTER_FEATS[_FEAT_TYPE]:
      if isnan(voter[feature]):
        example.append(avg_user[feature]) 
      else:
        example.append(voter[feature]) 
    av = (author['id'], voter['id'])
    u_sim = sim[av] if av in sim else avg_sim
    for feature in _SIM_FEATS[_FEAT_TYPE]:
      if isnan(u_sim[feature]):
        example.append(avg_sim[feature]) 
      else:
        example.append(u_sim[feature]) 
    u_conn = conn[av] if av in conn else avg_conn
    for feature in _CONN_FEATS[_FEAT_TYPE]:
      if isnan(u_conn[feature]):
        example.append(avg_conn[feature]) 
      else:
        example.append(u_conn[feature]) 
    X.append(example)
    y.append(vote['vote'])
    qid.append(vote['voter'])
  return X, y, qid


def predict():
  """ Predicts votes by applying a regressor technique.

      Args:
        None.

      Returns:
        None.
  """
  load_args()
  
  for i in xrange(NUM_SETS):
    for j in xrange(REP):
      print 'Reading data'
      reviews = load(open('%s/reviews-%d.pkl' % (_PKL_DIR, i), 'r'))
      users = load(open('%s/users-%d.pkl' % (_PKL_DIR, i), 'r'))
      train = load(open('%s/train-%d.pkl' % (_PKL_DIR, i), 'r'))
      test = load(open('%s/test-%d.pkl' % (_PKL_DIR, i), 'r'))
      val = load(open('%s/validation-%d.pkl' % (_PKL_DIR, i), 'r'))
      sim = load(open('%s/sim-%d.pkl' % (_PKL_DIR, i), 'r'))
      conn = load(open('%s/conn-%d.pkl' % (_PKL_DIR, i), 'r'))
   
      if _BIAS:
        pass #TODO

      avg_user = compute_avg_user(users)
      avg_sim = compute_avg_model(sim)
      avg_conn = compute_avg_model(conn)
      X_train, y_train, qid_train = generate_input(reviews, users, sim, conn,
          train, avg_user, avg_sim, avg_conn)
      X_val, _, qid_val = generate_input(reviews, users, sim, conn, val, avg_user,
          avg_sim, avg_conn)
      X_test, _, qid_test = generate_input(reviews, users, sim, conn, test, 
          avg_user, avg_sim, avg_conn)

      if _SCALE == 'all':
        scaler = fit_scaler('minmax', X_train)
        X_train = scale_features(scaler, X_train)
        X_val = scale_features(scaler, X_val)
        X_test = scale_features(scaler, X_test)
      else:
        qid_dep_size = len(sim.itervalues().next()) + len(conn.itervalues().next())
        scaler = fit_scaler_by_query('minmax', X_train, qid_train, qid_dep_size)
        X_train = scale_features(scaler, X_train, qid_train, qid_dep_size)
        X_val = scale_features(scaler, X_val, qid_val, qid_dep_size)
        X_test = scale_features(scaler, X_test, qid_test, qid_dep_size)

      model = GradientBoostingRegressor(loss=_LOSS, learning_rate=_ALPHA,
          n_estimators=_T, max_depth=_MAX_D, subsample=_SUBSAMPLE, 
          max_features=_MAX_F)
      model.fit(X_train , y_train)
      
      pred = model.predict(X_train)
      truth = [v['vote'] for v in train]
      print '~ Training error on set %d repetition %d' % (i, j)
      print 'RMSE: %f' % calculate_rmse(pred, truth)
      print 'nDCG@%d: %f' % (RANK_SIZE, calculate_avg_ndcg(train, reviews, pred,
          truth, RANK_SIZE))

      pred = model.predict(X_val)
      output = open('%s/gbrt-r:%f,t:%d,d:%d,l:%s,p:%f,m:%d,s:%s,f:%s,b:%s-%d-%d.dat' % 
          (_VAL_DIR, _ALPHA, _T, _MAX_D, _LOSS, _SUBSAMPLE, _MAX_F, _SCALE, 
          FEAT_TYPE, 'y' if _BIAS else 'n', i, j), 'w')
      for p in pred:
        print >> output, p
      output.close()
      
      pred = model.predict(X_test)
      output = open('%s/gbrt-r:%f,t:%d,d:%d,l:%s,p:%f,m:%d,s:%s,f:%s,b:%s-%d-%d.dat' % 
          (_OUTPUT_DIR, _ALPHA, _T, _MAX_D, _LOSS, _SUBSAMPLE, _MAX_F, _SCALE, 
          FEAT_TYPE, 'y' if _BIAS else 'n', i, j), 'w')
      for p in pred:
        print >> output, p
      output.close()


if __name__ == '__main__':
  predict()
