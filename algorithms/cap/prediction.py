""" Prediction Module for CAP
    -------------------------

    This module implements a Monte Carlo EM algorithm for fitting latent
    variables and data distributions' parameters.

    Usage:
    $ python -m algorithms.cap.prediction [-k <latent_dimensions>]
      [-e <em_iterations>] [-s <samples>] [-b <burn_in>] [-n <nr_iterations>]
      [-t <nr_tolerance>] [-l <nr_learning_rate>] [-a <eta>]
    where
    <latent_dimensions> is an integer with the number of latent dimensions,
    <iterations> is an integer with the maximum number of iterations of gradient
      descent.
    <em_iterations> is an integer with number of EM iterations,
    <samples> is an integer with number of gibbs samples in each EM iteration,
    <burn_in> is an integer with number of first samples ignored in Gibbs
      Sampling,
    <nr_iterations> is an integer with number of newton-raphson iterations,
    <nr_tolerant> is a float with newton-raphson convergence tolerance,
    <nr_learning_rate> is a float with newton-raphson learning rate,
    <eta> is a float constant used in OLS for easier inversion.
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
from algorithms.cap.map_features import map_features 
from algorithms.const import NUM_SETS, RANK_SIZE, REP 
from evaluation.metrics import calculate_rmse, calculate_avg_ndcg
from util.avg_model import compute_avg_user, compute_avg_model
from util.scaling import fit_scaler, scale_features


_OUTPUT_DIR = 'out/pred'
_VAL_DIR = 'out/val'
_PKL_DIR = 'out/pkl'
_K = 5
_EM_ITER = 10
_SAMPLES = 100
_BURN_IN = 10
_NR_ITER = 50
_NR_TOL = 1e-6
_NR_STEP = 0.1


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
      const.K = int(argv[i+1])
    elif argv[i] == '-e':
      const.EM_ITER = int(argv[i+1])
    elif argv[i] == '-s':
      const.SAMPLES = int(argv[i+1])
    elif argv[i] == '-b':
      const.BURN_IN = int(argv[i+1])
    elif argv[i] == '-n':
      const.NR_ITER = int(argv[i+1])
    elif argv[i] == '-t':
      const.NR_TOL = float(argv[i+1])
    elif argv[i] == '-l':
      const.NR_STEP = float(argv[i+1])
    elif argv[i] == '-a':
      const.ETA = float(argv[i+1])
    else:
      print ('Usage: $ python -m algorithms.cap.prediction'
          '[-k <latent_dimensions>] [-e <em_iterations>] [-s <samples>]'
          '[-b <burn_in>] [-n <nr_iterations>] [-t <nr_tolerance>]'
          '[-l <nr_learning_rate>] [-a <eta>]')
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


def fit_cap_scaler(features):
  """ Fit scaler for CAP, one per set of features.

      Args:
        features: dictionary of features indexed by entity name and containing a
      list of feature arrays.

      Returns:
        A dictionary of scalers, indexed by entity name.
  """
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
  """ Scales features for CAP using previously fitted scaler.

      Args:
        scaler: dictionary of scalers, indexed by entity name.
        features: dictionary of list of feature arrays, indexed by entity name.

      Returns:
        A new dictionary of features, in the same format but with scaled values.
  """
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
        train: list of votes in training set.
        users: dictionary o users. 
        trusts: networkx DiGraph with trust network. 
        features: dictionary of a list of feature arrays, indexed by entity or
      interaction id and containing features for each vote in training.

      Returns:
        The same dictionary of Group objects with instances added.
  """
  sim_i = 0
  conn_i = 0
  for i, vote in enumerate(train):
    # features are added in the same order of ids votes in train
    r_id, a_id, v_id = vote['review'], vote['author'], vote['voter']
    var_groups['alpha'].add_instance(v_id, features['voter'][i], train)
    var_groups['beta'].add_instance(r_id, features['review'][i], train) 
    var_groups['xi'].add_instance(a_id, features['author'][i], train)
    var_groups['u'].add_instance(v_id, features['voter'][i], train) 
    var_groups['v'].add_instance(r_id, features['review'][i], train) 
    if v_id in users and a_id in users[v_id]['similars']:
      var_groups['gamma'].add_instance((a_id, v_id), features['sim'][sim_i],
          train)
      sim_i += 1
    if v_id in trusts and a_id in trusts[v_id]:
      var_groups['lambda'].add_instance((a_id, v_id), features['conn'][conn_i],
          train)
      conn_i += 1
  return var_groups


def calculate_predictions(groups, test, users, trusts, features, sim, conn):
  """ Calculates the predictions after fitting values. If the vote to be
      predicted contains entities modeled as latent variables (i.e., present
      on training set), the latent variable is used; otherwise, it is
      approximated by linear regression over features.

      Args:
        groups: dictionary of Group objects.
        test: list of vote dictionaries on test set.
        users: dictionary of user dictionaries.
        trusts: networkx DiGraph with trust network. 
        features: dictionary of a list of feature arrays, indexed by entity or
      interaction id and containing features for each vote in training.
        sim: dictionary of similarity of users dictionaries.
        conn: dictionary of connection of users dictionaries.

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
    if v_id in users and a_id in users[v_id]['similars'] and (a_id, v_id) in sim:
      sim_feat = features['sim'][sim_i]
      sim_feat = sim_feat.reshape((sim_feat.size, 1))
      gamma = groups['gamma'].get_instance(vote).value if \
          groups['gamma'].contains(vote) else \
          groups['gamma'].weight_param.value.T.dot(sim_feat)[0,0]
      sim_i += 1
    lambd = 0.0
    if v_id in trusts and a_id in trusts[v_id] and (a_id, v_id) in conn:
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
  
  for i in xrange(NUM_SETS):
    print 'Reading data'
    reviews = load(open('%s/reviews-%d.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
    users = load(open('%s/users-%d.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
    train = load(open('%s/train-%d.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
    test = load(open('%s/test-%d.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
    val = load(open('%s/validation-%d.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
    trusts = load(open('%s/trusts.pkl' % _PKL_DIR, 'r'))
    sim = load(open('%s/sim-%d.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
    conn = load(open('%s/conn-%d.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
    f_train = map_features(train, reviews, users, sim, conn, trusts)
    f_val = map_features(val, reviews, users, sim, conn, trusts)
    f_test = map_features(test, reviews, users, sim, conn, trusts)
    for j in xrange(REP):
      print 'Creating variables'
     # scaler = fit_cap_scaler(f_train)
     # f_train = scale_cap_features(scaler, f_train)
      var_groups = create_variable_groups()
      populate_variables(var_groups, train, users, trusts, f_train)
      print 'Running EM'
      expectation_maximization(var_groups, train)
      print 'Calculating Predictions'
      pred = calculate_predictions(var_groups, train, users, trusts, f_train, sim,
          conn)
      print 'TRAINING ERROR'
      truth = [v['vote'] for v in train]
      print '-- RMSE: %f' % calculate_rmse(pred, truth) 
      print '-- nDCG@%d: %f' % (RANK_SIZE, calculate_avg_ndcg(train, reviews,
          pred, truth, RANK_SIZE)
      print 'Outputting Validation Prediction'
     # f_val = scale_cap_features(scaler, f_val)
      pred = calculate_predictions(var_groups, val, users, trusts, f_val, sim, conn)
      output = open('%s/cap-%d-%d.dat' % (_VAL_DIR, i, j), 'w')
      for p in pred:
        print >> output, p
      output.close()
      print 'Outputting Test Prediction'
     # f_test = scale_cap_features(scaler, f_test)
      pred = calculate_predictions(var_groups, test, users, trusts, f_test, sim,
          conn)
      output = open('%s/cap-%d-%d.dat' % (_OUTPUT_DIR, i, j), 'w')
      for p in pred:
        print >> output, p
      output.close()

  


if __name__ == '__main__':
  run() 
