""" RLFM Module
    -----------

    Applies a Regression-Based Latent Factor Model (RLFM), a recommender
    technique based on linear combination of latent factors, whose distribution
    is centered on the regression of observed features.

    Usage:
      $ python -m algo.recsys.rlfm [-k <k>] [-i <iterations>] [-g <gibbs_samples>]
        [-b <burn_in>] [-s <scale>] [-f <feature_set>]
    where:
    <k> is the number of latent dimensions,
    <iterations> is the number of EM iterations of the method,
    <gibbs_samples> is an integer with number of gibbs samples in each EM 
      iteration,
    <burn_in> is the number of initial discarded samples in each sampling,
    <feature_set> is in the set ['www', 'cap', 'all'].
"""


from commands import getstatusoutput, getoutput
from pickle import dump, load
from numpy import array, nan, isnan, vstack
from sys import argv

from algo.const import NUM_SETS, RANK_SIZE, REP, REVIEW_FEATS, \
    AUTHOR_FEATS, VOTER_FEATS, SIM_FEATS, CONN_FEATS 
from perf.metrics import calculate_rmse, calculate_avg_ndcg
from util.avg_model import compute_avg_user, compute_avg_model
from util.scaling import fit_scaler, scale_features 


_K = 5
_ITER = 10
_SAMPLES = 100
_BURN_IN = 10
_FEAT = 'cap'
_DATA_DIR = 'out/data'
_TRAIN_DIR = 'out/train'
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
    elif argv[i] == '-g':
      global _SAMPLES
      _SAMPLES = int(argv[i+1])
    elif argv[i] == '-b':
      global _BURN_IN
      _BURN_IN = int(argv[i+1])
    elif argv[i] == '-f' and argv[i+1] in ['www', 'cap', 'all']:
      global _FEAT
      _FEAT = argv[i+1]
    else:
      print ('Usage: python -m algo.recsys.rlfm [-k <k>] [-i <iterations>] '
          '[-g <gibb_samples>] [-b <burn_in>] [-f <feature_type>]')
      exit()
    i = i + 2
  global _CONF_STR
  _CONF_STR = 'k:%d,i:%d,g:%d,b:%d,f:%s' % (_K, _ITER, _SAMPLES, _BURN_IN,
      _FEAT)


def model_dyad(votes, sim, conn, avg_sim, avg_conn):
  """ Models a dyad observed data (interaction features) for each vote.

      Args;
        votes: list of models whose instances have to be modeled.
        sim: dictionary of similarity dictionaries.
        conn: dictionary of connection dictionaries.
        avg_sim: dictionary with average similarity for mean imputation.
        avg_conn: dictionary with average connection for mean imputation.

      Returns:
        An arras of arrays with observed features of each votes.
  """
  model = []
  for vote in votes:
    instance = []
    av = vote['author'], vote['voter']
    u_sim = sim[av] if av in sim else avg_sim
    for feature in SIM_FEATS[_FEAT]:
      if isnan(u_sim[feature]):
        instance.append(avg_sim[feature])
      else:
        instance.append(u_sim[feature])
    u_conn = conn[av] if av in conn else avg_conn
    for feature in CONN_FEATS[_FEAT]:
      if isnan(u_conn[feature]):
        instance.append(avg_conn[feature])
      else:
        instance.append(u_conn[feature])
    model.append(instance)
  return array(model)


def output_dyad(name, votes, model, i):
  """ Outputs a dyadic model to file, using RLFM library standards.

      Args:
        name: name of the set of votes, either train, val of test.
        votes: list of votes to output.
        model: list of observed features.
        i: set index related to this data.

      Returns:
        None. Data is printed to proper files.
  """
  outfile = open('%s/rlfm_%s-%s-%d.dat' % (_DATA_DIR, name, _CONF_STR, i), 'w')
  obs = open('%s/rlfm_%s_obs-%s-%d.dat' % (_DATA_DIR, name, _CONF_STR, i), 'w')
  print >> outfile, 'src_id\tdst_id\ty'
  for i, vote in enumerate(votes):
    voter_id = vote['voter']
    review_id = vote['review']
    print >> outfile, '%s\t%d\t%d ' % (voter_id, review_id, vote['vote'])
    line = ''
    instance = model[i]
    for value in instance:
      line += '%f\t' % value
    print >> obs, line.strip()
  outfile.close()
  obs.close()


def model_items(reviews, users, train_reviews, test_reviews, avg_user):
  """ Models items (dst in RLFM) in two sets, train and test.
  
      Observations:
      - The division is made to allow scaling using only train, but the final
      model is a single one.

      Args:
        reviews: dictionary of reviews' dictionaries.
        users: dictionary of users' dictionaries.
        train_reviews: set of review ids which belong in train.
        test_reviews: set of review ids which belong in test.
        avg_user: dictionary with average user model for mean imputation.

      Returns:
        Four array-like: with arrays of features for items in train, with ids of
      each item in previous array, with arrays of features for items in test,
      with ids for each item in previous array.
  """
  train_model = []
  train_key = []
  test_model = []
  test_key = []
  for review_id in train_reviews:
    review = reviews[review_id]
    instance = []
    for feature in REVIEW_FEATS[_FEAT]: 
      instance.append(review[feature]) 
    author = users[review['author']] if review['author'] in users else avg_user
        # author features are associated to item
    for feature in AUTHOR_FEATS[_FEAT]:
      if isnan(author[feature]):
        instance.append(avg_user[feature]) 
      else:
        instance.append(author[feature])
    train_model.append(instance)
    train_key.append(review_id)
  for review_id in test_reviews:
    if review_id in train_reviews:
      continue
    review = reviews[review_id]
    instance = []
    for feature in REVIEW_FEATS[_FEAT]: 
      instance.append(review[feature]) 
    author = users[review['author']] if review['author'] in users else avg_user
        # author features are associated to item
    for feature in AUTHOR_FEATS[_FEAT]:
      if isnan(author[feature]):
        instance.append(avg_user[feature]) 
      else:
        instance.append(author[feature])
    test_model.append(instance)
    test_key.append(review_id)
  return array(train_model), train_key, array(test_model), test_key


def model_users(users, train_users, test_users, avg_user):
  """ Models users (src in RLFM) in two sets, train and test.
  
      Observations:
      - The division is made to allow scaling using only train, but the final
      model is a single one.

      Args:
        users: dictionary of users' dictionaries.
        train_users: set of user ids which belong in train.
        test_users: set of user ids which belong in test.
        avg_user: dictionary with average user model for mean imputation.

      Returns:
        Four array-like: with arrays of features for users in train, with ids of each
      user in previous array, with array of features for users in test, with ids
      for each user in previous array.
  """
  train_model = []
  train_key = []
  test_model = []
  test_key = []
  for user_id in train_users:
    instance = []
    user = users[user_id] if user_id in users else avg_user
    for feature in VOTER_FEATS[_FEAT]:
      if isnan(user[feature]):
        instance.append(avg_user[feature]) 
      else:
        instance.append(user[feature]) 
    train_model.append(instance)
    train_key.append(user_id)
  for user_id in test_users:
    if user_id in train_users:
      continue
    instance = []
    user = users[user_id] if user_id in users else avg_user
    for feature in VOTER_FEATS[_FEAT]:
      if isnan(user[feature]):
        instance.append(avg_user[feature]) 
      else:
        instance.append(user[feature]) 
    test_model.append(instance)
    test_key.append(user_id)
  return array(train_model), train_key, array(test_model), test_key


def output_entity(name, model, ids, i):
  """ Outputs an entity, either item or user, to file in proper RLFM library
      format.

      Args:
        name: string with the name of the entity.
        model: list of features' lists representing each entity.
        ids: list of entity ids, in the same order as model.
        i: index of data split being used.

      Returns:
        None. Data is output to proper file.
  """
  efeat = open('%s/rlfm_%s-%s-%d.dat' % (_DATA_DIR, name, _CONF_STR, i), 'w')
  for i in xrange(len(model)):
    line = str(ids[i])
    for value in model[i]:
      line += '\t%f' % value
    print >> efeat, line
  efeat.close()


def main():
  """ Main method, which performs prediction and outputs to file.

      Args:
        None.

      Returns:
        None.
  """
  load_args()
  
  for i in xrange(NUM_SETS):
    print 'Reading data'
    reviews = load(open('%s/reviews-%d.pkl' % (_PKL_DIR, i), 'r'))
    users = load(open('%s/users-%d.pkl' % (_PKL_DIR, i), 'r'))
    train = load(open('%s/train-%d.pkl' % (_PKL_DIR, i), 'r'))
    test = load(open('%s/test-%d.pkl' % (_PKL_DIR, i), 'r'))
    val = load(open('%s/validation-%d.pkl' % (_PKL_DIR, i), 'r'))
    sim = load(open('%s/sim-%d.pkl' % (_PKL_DIR, i), 'r'))
    conn = load(open('%s/conn-%d.pkl' % (_PKL_DIR, i), 'r'))
    
    print 'Creating average user (for mean imputation)'
    avg_user = compute_avg_user(users)
    avg_sim = compute_avg_model(sim)
    avg_conn = compute_avg_model(conn)
    
    print 'Modeling'
    X_train = model_dyad(train, sim, conn, avg_sim, avg_conn)
    X_val = model_dyad(val, sim, conn, avg_sim, avg_conn)
    X_test = model_dyad(test, sim, conn, avg_sim, avg_conn)
    train_reviews = set([v['review'] for v in train])
    test_reviews = set([v['review'] for v in val]).union(set([v['review'] for v
        in test]))
    X_item_train, item_train_key , X_item_test, item_test_key = \
        model_items(reviews, users, train_reviews, test_reviews, avg_user)
        # train, test: same file, different scaling
    train_users = set([v['voter'] for v in train])
    test_users = set([v['voter'] for v in val]).union(set([v['voter'] for v in
        test]))
    X_user_train, user_train_key, X_user_test, user_test_key = \
        model_users(users, train_users, test_users, avg_user)

    print 'Scaling'
    dyad_scaler = fit_scaler('minmax', X_train)
    X_train = scale_features(dyad_scaler, X_train)
    X_val = scale_features(dyad_scaler, X_val)
    X_test = scale_features(dyad_scaler, X_test)
    item_scaler = fit_scaler('minmax', X_item_train)
    X_item_train = scale_features(item_scaler, X_item_train)
    X_item_test = scale_features(item_scaler, X_item_test)
    user_scaler = fit_scaler('minmax', X_user_train)
    X_user_train = scale_features(user_scaler, X_user_train)
    X_user_test = scale_features(user_scaler, X_user_test)
    X_item = vstack((X_item_train, X_item_test))
    item_key = item_train_key + item_test_key
    X_user = vstack((X_user_train, X_user_test))
    user_key = user_train_key + user_test_key

    print 'Outputting model'
    output_dyad('train', train, X_train, i)
    output_dyad('val', val, X_val, i)
    output_dyad('test', test, X_test, i)
    output_entity('item', X_item, item_key, i)
    output_entity('user', X_user, user_key, i)

    for j in xrange(REP):
      print 'Fitting model'
      print getoutput(('Rscript lib/rlfm/rlfm_fit.R %d %d %d %d %s %d %d '
          '%s') % (_K, _ITER, _SAMPLES, _BURN_IN, _FEAT, i, j, _DATA_DIR))

      print getoutput('Rscript lib/rlfm/rlfm_predict.R %d %d %d %d %s %d %d '
          '%s train' % (_K, _ITER, _SAMPLES, _BURN_IN, _FEAT, i, j, _DATA_DIR))

      predfile = open('%s/rlfm-%s-%d-%d.dat' % (_TRAIN_DIR, _CONF_STR, i,
          j), 'r')
      pred = [float(p.strip()) for p in predfile]
      predfile.close()
      truth = [v['vote']  for v in train]
      print len(pred)
      print len(truth)
      print '~ Training error on set %d repetition %d' % (i, 0)
      print 'RMSE: %f' % calculate_rmse(pred, truth)
      print 'nDCG@%d: %f' % (RANK_SIZE, calculate_avg_ndcg(train, reviews, pred,
          truth, RANK_SIZE))

      print 'Predicting in validation'
      print getoutput('Rscript lib/rlfm/rlfm_predict.R %d %d %d %d %s %d %d '
          '%s val' % (_K, _ITER, _SAMPLES, _BURN_IN, _FEAT, i, j, _DATA_DIR))

      print 'Predicting in test'
      print getoutput('Rscript lib/rlfm/rlfm_predict.R %d %d %d %d %s %d %d '
          '%s test' % (_K, _ITER, _SAMPLES, _BURN_IN, _FEAT, i, j, _DATA_DIR))


if __name__ == '__main__':
  main() 
