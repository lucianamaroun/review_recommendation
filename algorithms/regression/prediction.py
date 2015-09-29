""" Prediction using Regression Methods
    -----------------------------------

    Use of regression methods for predicting relevance of reviews for users.

    Usage:
      $ python -m script.ml [-s <sample_size>] [-p <predictor>]
    where <sample_size> is a float with the fraction of reviews contained in the
    sample and <predictor> is a string with the prediction name, which can be
    one of "svr", "lr" or "gbrt".
"""


from pickle import load
from sys import argv, exit
from time import time

from numpy import nan, isnan
from numpy.random import seed
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from evaluation.metrics import calculate_rmse, calculate_ndcg
from util.avg_model import compute_avg_user, compute_avg_model 
from util.scaling import fit_scaler, fit_scaler_by_query, scale_features

seed(int(time()*1000000)%1000000)
_PREDICTORS = {'svr': lambda: SVR(C=10, epsilon=0.01, kernel='rbf'),
    'lr': lambda: Ridge(alpha=1000),
    'gbrt': lambda: GradientBoostingRegressor(n_estimators=100,
      learning_rate=0.01, max_depth=3,
      random_state=int(time()*1000000)%1000000)
}
_SAMPLE = 0.05
_OUTPUT_DIR = 'out/pred'
_VAL_DIR = 'out/val'
_PKL_DIR = 'out/pkl'
_PRED = 'lr'
_FEAT_TYPE = 'cap'
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
    if argv[i] == '-s':
      global _SAMPLE
      _SAMPLE = float(argv[i+1])
    elif argv[i] == '-p' and argv[i+1] in _PREDICTORS:
      global _PRED 
      _PRED = argv[i+1]
    else:
      print ('Usage: python -m methods.regression.prediction [-s <sample_size>]'
          '[-p <predictor>], <predictor> is in the set [svr, rfr, lr, mart]')
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
   
  reviews = load(open('%s/reviews%.2f.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
  users = load(open('%s/users%.2f.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
  train = load(open('%s/train%.2f.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
  validation = load(open('%s/validation%.2f.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
  test = load(open('%s/test%.2f.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
  sim = load(open('%s/sim%.2f.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
  conn = load(open('%s/conn%.2f.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
  
  avg_user = compute_avg_user(users)
  avg_sim = compute_avg_model(sim)
  avg_conn = compute_avg_model(conn)
  X_train, y_train, qid_train = generate_input(reviews, users, sim, conn, train,
      avg_user, avg_sim, avg_conn)
  X_val, _, qid_val = generate_input(reviews, users, sim, conn, validation,
      avg_user, avg_sim, avg_conn)
  X_test, _, qid_test = generate_input(reviews, users, sim, conn, test,
      avg_user, avg_sim, avg_conn)

  qid_dep_size = len(sim.itervalues().next()) + len(conn.itervalues().next())
 # scaler = fit_scaler('minmax', X_train)
 # X_train = scale_features(scaler, X_train)
 # X_val = scale_features(scaler, X_val)
 # X_test = scale_features(scaler, X_test)
  scaler = fit_scaler_by_query('minmax', X_train, qid_train, qid_dep_size)
  X_train = scale_features(scaler, X_train, qid_train, qid_dep_size)
  X_val = scale_features(scaler, X_val, qid_val, qid_dep_size)
  X_test = scale_features(scaler, X_test, qid_test, qid_dep_size)

  clf = _PREDICTORS[_PRED]()
  clf.fit(X_train , y_train)
  
  pred = clf.predict(X_train)
  truth = [v['vote'] for v in train]
  print 'TRAINING ERROR'
  print 'RMSE: %f' % calculate_rmse(pred, truth)
  pred_group = {}
  truth_group = {}
  for i in xrange(len(train)):
    voter = train[i]['voter']
    if voter not in pred_group:
      pred_group[voter] = []
      truth_group[voter] =[]
    pred_group[voter].append(pred[i])
    truth_group[voter].append(train[i]['vote'])
  for i in xrange(5, 21, 5):
    score_sum = 0.0
    for key in pred_group:
      score_sum += calculate_ndcg(pred_group[key], truth_group[key], i)
    score = score_sum / len(pred_group)
    print 'NDCG@%d: %f' % (i, score)

  pred = clf.predict(X_val)
  output = open('%s/%s%.2f.dat' % (_VAL_DIR, _PRED, _SAMPLE * 100), 'w')
  for p in pred:
    print >> output, p
  output.close()
  
  pred = clf.predict(X_test)
  output = open('%s/%s%.2f.dat' % (_OUTPUT_DIR, _PRED, _SAMPLE * 100), 'w')
  for p in pred:
    print >> output, p
  output.close()


if __name__ == '__main__':
  predict()
