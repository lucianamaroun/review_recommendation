""" Prediction Module for CAP
    -------------------------

    This module implements a Monte Carlo EM algorithm for fitting latent
    variables and data distributions' parameters.

    Usage:
      $ python -m algorithms.cap.prediction [-s <sample_size>] [-k <k>]
    where <sample_size> is a float with the fraction of the sample and K is an
    integer with the number of latent factor dimensions.
"""

from math import sqrt
from sys import argv, exit

from numpy import zeros, isnan
from pickle import load

from algorithms.cap.models import EntityScalarGroup, EntityArrayGroup, \
    InteractionScalarGroup, EntityScalarParameter, EntityArrayParameter, \
    InteractionScalarParameter, ScalarVarianceParameter, \
    ArrayVarianceParameter, PredictionVarianceParameter
from algorithms.cap import const
from algorithms.cap.em import expectation_maximization
from algorithms.cap.map_features import map_review_features, map_author_features, \
    map_voter_features, map_users_sim_features, map_users_conn_features
from evaluation.metrics import calculate_rmse, calculate_ndcg
from util.avg_model import compute_avg_user, compute_avg_model
from util.scaling import fit_scaler, scale_features


_SAMPLE = 0.001
_AVG_USER = None
_AVG_SIM = None
_AVG_CONN = None
_OUTPUT_DIR = 'out/pred'
_VAL_DIR = 'out/val'
_PKL_DIR = 'out/pkl'


def load_args():
  """ Loads command line arguments.

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
      const.K = int(argv[i+1])
    else:
      print 'Usage: python -m algorithms.cap.prediction [-s <sample_size>] [-k <k>]'
      exit()
    i = i + 2


def create_variable_groups():
  """ Creates empty latent variable groups with its corresponding parameters.

      Args:
        None.
      
      Returns:
        A dictionary of Group objects indexed by names.
  """
  var_H = PredictionVarianceParameter('var_H')
  var_groups = {
    'alpha': EntityScalarGroup('alpha', 'voter', 
        EntityScalarParameter('d', (9, 1)), 
        ScalarVarianceParameter('var_alpha'), var_H),
    'beta': EntityScalarGroup('beta', 'review', 
        EntityScalarParameter('g', (17, 1)), 
        ScalarVarianceParameter('var_beta'), var_H),
    'xi': EntityScalarGroup('xi', 'author', 
        EntityScalarParameter('b', (5, 1)), 
        ScalarVarianceParameter('var_xi'), var_H),
    'u': EntityArrayGroup('u', (const.K, 1), 'voter', 
        EntityArrayParameter('W', (const.K, 9)),
        ArrayVarianceParameter('var_u'), var_H),
    'v': EntityArrayGroup('v', (const.K, 1), 'review', 
        EntityArrayParameter('V', (const.K, 17)),
        ArrayVarianceParameter('var_v'), var_H),
    'gamma': InteractionScalarGroup('gamma', ('author', 'voter'),
        InteractionScalarParameter('r', (7, 1)), 
          ScalarVarianceParameter('var_gamma'), var_H),
    'lambda': InteractionScalarGroup('lambda', ('author', 'voter'),
        InteractionScalarParameter('h', (5, 1)), 
        ScalarVarianceParameter('var_lambda'), var_H)
  }
  var_groups['u'].set_pair_name('v')
  var_groups['v'].set_pair_name('u')
  return var_groups

# TODO: put inside map features
def map_features(votes, reviews, users, users_sim, users_conn, trusts):
  global _AVG_USER
  _AVG_USER = _AVG_USER if _AVG_USER else compute_avg_user(users)
  global _AVG_SIM
  _AVG_SIM = _AVG_SIM if _AVG_SIM else compute_avg_model(users_sim)
  global _AVG_CONN
  _AVG_CONN = _AVG_CONN if _AVG_CONN else compute_avg_model(users_conn)
  features = {'review': [], 'author': [], 'voter': [], 'sim': [], 'conn': []}
  for vote in votes:
    r_id, a_id, v_id = vote['review'], vote['author'], vote['voter']
    r_feat = map_review_features(reviews[r_id])
    features['review'].append(r_feat)
    author = users[a_id] if a_id in users else _AVG_USER
    a_feat = map_author_features(author, _AVG_USER)
    features['author'].append(a_feat)
    voter = users[v_id] if v_id in users else _AVG_USER
    v_feat = map_voter_features(voter, _AVG_USER)
    features['voter'].append(v_feat)
    if v_id in users and a_id in users[v_id]['similars']:
      sim = users_sim[(a_id, v_id)] if (a_id, v_id) in users_sim else _AVG_SIM
      sim_feat = map_users_sim_features(sim, _AVG_SIM)
      features['sim'].append(sim_feat)
    if v_id in trusts and a_id in trusts[v_id]:
      conn = users_conn[(a_id, v_id)] if (a_id, v_id) in users_conn else \
          _AVG_CONN
      conn_feat = map_users_conn_features(conn, _AVG_CONN)
      features['conn'].append(conn_feat)
  return features


def fit_cap_scaler(features):
  scaler = {'review': None, 'author': None, 'voter': None, 'sim': None,
      'conn': None}
  scaler['review'] = fit_scaler('minmax', features['review']) 
  scaler['author'] = fit_scaler('minmax', features['author']) 
  scaler['voter'] = fit_scaler('minmax', features['voter']) 
  if features['sim']:
    scaler['sim'] = fit_scaler('minmax', features['sim'])
  if features['conn']:
    scaler['conn'] = fit_scaler('minmax', features['conn']) 
  return scaler


def scale_cap_features(scaler, features):
  features['review'] = scale_features(scaler['review'], features['review'])
  features['author'] = scale_features(scaler['author'], features['author'])
  features['voter'] = scale_features(scaler['voter'], features['voter'])
  if features['sim']:
    features['sim'] = scale_features(scaler['sim'], features['sim'])
  if features['conn']:
    features['conn'] = scale_features(scaler['conn'], features['conn'])
  return features


def populate_variables(var_groups, train, users, trusts, features):
  """ Populates the dictionary of Group objects by adding instances
      related to an entity.
  
      Args:
        var_groups: a dictionary of Group objects indexed by group names.
        reviews: dictionary of reviews' features indexed by id.
        authors: dictionary of authors' features indexed by id.
        votes: dictionary of votes' features indexed by id.
        trusts: networkx DiGraph with trust network. 
        users_sim: dictionary of users similarity features indexed by a tuple
          (author_id, voter_id).
        users_conn: dictionary of users connection features indexed by a tuple
          (author_id, voter_id).

      Returns:
        The same dictionary of Group objects with instances added.
  """
  sim_i = 0
  conn_i = 0
  for i, vote in enumerate(train):
    # features are added in the same order of votes in train
    r_id, a_id, v_id = vote['review'], vote['author'], vote['voter']
    var_groups['alpha'].add_instance(v_id, features['voter'][i])
    var_groups['beta'].add_instance(r_id, features['review'][i]) 
    var_groups['xi'].add_instance(a_id, features['author'][i])
    var_groups['u'].add_instance(v_id, features['voter'][i]) 
    var_groups['v'].add_instance(r_id, features['review'][i]) 
    if v_id in users and a_id in users[v_id]['similars']:
      var_groups['gamma'].add_instance((a_id, v_id), features['sim'][sim_i])
      sim_i += 1
    if v_id in trusts and a_id in trusts[v_id]:
      var_groups['lambda'].add_instance((a_id, v_id), features['conn'][conn_i])
      conn_i += 1
  return var_groups


def calculate_predictions(groups, test, users, trusts, features):
  """ Calculate the predictions after fitting values. If the vote to be
      predicted contains entities modeled as latent variables (i.e., present
      on training set), the latent variable is used; otherwise, it is
      approximated by linear regression over features.

      Args:
        groups: dictionary of Group objects.
        test: list of vote dictionaries on test set.
        reviews: dictionary of review dictionaries.
        users: dictionary of user dictionaries.
        users_sim: dictionary of similarity of users dictionaries.
        users_conn: dictionary of connection of users dictionaries.

      Returns:
        A list of floats containing prediction values for each vote in test, in
      the same order.
  """
  pred = []
  ignored = 0
  sim_i = 0
  conn_i = 0
  for i, vote in enumerate(test):
    v_feat = features['voter'][i]
    v_feat = v_feat.reshape((v_feat.size, 1))
    alfa = groups['alpha'].get_instance(vote).value if \
          groups['alpha'].contains(vote) else groups['alpha'].weight_param.value.T \
          .dot(v_feat)[0,0]
    u = groups['u'].get_instance(vote).value if groups['u'].contains(vote) \
        else groups['u'].weight_param.value.dot(v_feat)
    r_feat = features['review'][i]
    r_feat = r_feat.reshape((r_feat.size, 1))
    beta = groups['beta'].get_instance(vote).value if \
        groups['beta'].contains(vote) else groups['beta'].weight_param.value.T \
        .dot(r_feat)[0,0]
    v = groups['v'].get_instance(vote).value if groups['v'].contains(vote) \
        else groups['v'].weight_param.value.dot(r_feat)
    a_feat = features['author'][i]
    a_feat = a_feat.reshape((a_feat.size, 1))
    xi = groups['xi'].get_instance(vote).value if \
        groups['xi'].contains(vote) else groups['xi'].weight_param.value.T \
        .dot(a_feat)[0,0]
    a_id, v_id = vote['author'], vote['voter']
    gamma = 0.0
    if v_id in users and a_id in users[v_id]['similars']:
      sim_feat = features['sim'][sim_i]
      sim_feat = sim_feat.reshape((sim_feat.size, 1))
      gamma = groups['gamma'].get_instance(vote).value if \
          groups['gamma'].contains(vote) else \
          groups['gamma'].weight_param.value.T.dot(sim_feat)[0,0]
      sim_i += 1
    lambd = 0.0
    if v_id in trusts and a_id in trusts[v_id]:
      conn_feat = features['conn'][conn_i]
      conn_feat = conn_feat.reshape((conn_feat.size, 1))
      lambd = groups['lambda'].get_instance(vote).value if \
          groups['lambda'].contains(vote) else \
          groups['lambda'].weight_param.value.T.dot(conn_feat)[0,0]
      conn_i += 1
    prediction = u.T.dot(v)[0,0] + alfa + beta + xi + gamma + lambd
    pred.append(prediction)
  return pred


def run():
  load_args()
  
  print 'Reading pickles'
  reviews = load(open('%s/reviews%d.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
  users = load(open('%s/users%d.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
  train = load(open('%s/train%d.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
  test = load(open('%s/test%d.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
  val = load(open('%s/validation%d.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
  trusts = load(open('%s/trusts%d.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
  sim = load(open('%s/sim%d.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
  conn = load(open('%s/conn%d.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
  train_dict = {i:vote for i, vote in enumerate(train)}
  
  print 'Creating variables'
  f_train = map_features(train, reviews, users, sim, conn, trusts)
  #scaler = fit_cap_scaler(f_train)
  #f_train = scale_cap_features(scaler, f_train)
  var_groups = create_variable_groups()
  populate_variables(var_groups, train, users, trusts, f_train)
  
  print 'Running EM'
  expectation_maximization(var_groups, train_dict)

  print 'Calculating Predictions'
  pred = calculate_predictions(var_groups, train, users, trusts, f_train)
  print 'TRAINING ERROR'
  truth = [v['vote'] for v in train]
  rmse = calculate_rmse(pred, truth) 
  print 'RMSE: %s' % rmse
  for i in xrange(5, 21, 5):
    score = calculate_ndcg(pred, truth, i)
    print 'NDCG@%d: %f' % (i, score)

  print 'Outputting Validation Prediction'
  f_val = map_features(val, reviews, users, sim, conn, trusts)
  #f_val = scale_cap_features(scaler, f_val)
  pred = calculate_predictions(var_groups, val, users, trusts, f_val)
  output = open('%s/cap%.2f.dat' % (_VAL_DIR, _SAMPLE * 100), 'w')
  for p in pred:
    print >> output, p
  output.close()

  print 'Outputting Test Prediction'
  f_test = map_features(test, reviews, users, sim, conn, trusts)
  #f_test = scale_cap_features(scaler, f_test)
  pred = calculate_predictions(var_groups, test, users, trusts, f_test)
  output = open('%s/cap%.2f.dat' % (_OUTPUT_DIR, _SAMPLE * 100), 'w')
  for p in pred:
    print >> output, p
  output.close()


if __name__ == '__main__':
  run() 
