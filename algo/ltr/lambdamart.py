""" LambdaMart
    ----------

    Applies a LambdaMART, a pairwise learning to rank technique, for
    recommending reviews.

    Usage:
      $ python -m algo.ltr.lambdamart [-l <learning_rate>] [-n <leaves>]
        [-t <trees>] [-f <feature_set>] [-b <bias>]
    where:
    <learning_rate> is a float which update step in each stage,
    <leaves> is the number of leaves of each tree (2^depth - 1),
    <trees> is the number of trees to build the model with,
    <feature_set> is in the set ['www', 'cap', 'all'], and 
    <bias> is either 'y' or 'n'.
"""


from commands import getstatusoutput, getoutput
from sys import argv

from numpy import nan, isnan
from pickle import load

from algo.const import NUM_SETS, RANK_SIZE, REP, REVIEW_FEATS, AUTHOR_FEATS, \
    VOTER_FEATS, SIM_FEATS, CONN_FEATS 
from perf.metrics import calculate_rmse, calculate_avg_ndcg
from util.avg_model import compute_avg_user, compute_avg_model 
from util.bias import BiasModel
from util.scaling import fit_scaler, fit_scaler_by_query, scale_features


_ALPHA = 0.01
_L = 10
_T = 100
_FEAT_TYPE = 'cap'
_BIAS = False
_PKL_DIR = 'out/pkl'
_VAL_DIR = 'out/val'
_DATA_DIR = 'out/data'
_MODEL_DIR = 'out/model'
_OUTPUT_DIR = 'out/test'
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
    if argv[i] == '-l':
      global _ALPHA
      _ALPHA = float(argv[i+1])
    elif argv[i] == '-n':
      global _L
      _L = int(argv[i+1])
    elif argv[i] == '-t':
      global _T
      _T = int(argv[i+1])
    elif argv[i] == '-f' and argv[i+1] in ['www', 'cap', 'all']:
      global _FEAT_TYPE
      _FEAT_TYPE = argv[i+1]
    elif argv[i] == '-b' and argv[i+1] in ['y', 'n']:
      global _BIAS
      _BIAS = True if argv[i+1] == 'y' else False
    else:
      print ('Usage: \n'
          '$ python -m algo.ltr.lambdamart [-l <learning_rate>] [-n <leaves>] '
          '[-t <trees>] [-f <feature_set>] [-b <bias>]')
      exit()
    i = i + 2
  global _CONF_STR
  _CONF_STR = 'l:%f,n:%s,t:%s,f:%s,b:%s' % (_ALPHA, _L, _T, _FEAT_TYPE, 
      'y' if _BIAS else 'n')


def model_str(x, y, qid):
  """ Models features to a string, in the required format for SVMRank. The
      format consists in outputting the vote id, followed by the 'qid:<qid>'
      where <qid> is the query id and, then, by features in the format
      '<i>:<value>', where <i> is the index of the feature (dimension) and
      <value> is its value. Also, <qid> should be sorted over lines.

      Args:
        x: feature vector of the instance.
        y: truth value of the instance.
        qid: query id of the instance.

      Returns:
        A string with the line ready to insert in output file.
  """
  line = '%d qid:%s ' % (y, qid)
  count = 1
  for i in xrange(x.size):
    line += '%d:%f ' % (count, x[i])
    count += 1
  return line


def output_model(X, y, qid, outfile):
  """ Outputs the model to a file, with one vote in each line.

      Args:
        X: list of feature lists
        y: list of truth values (might be None, in which case it does not matter
      the truth, for validation and test)
        qid: list of query ids associated to each instance.
        outfile: file descriptior to output.

      Returns:
        None. Outputs to outfile.
  """
  index = range(len(X))
  index = sorted(index, key=lambda i: qid[i])
  for i in index:
    print >> outfile, model_str(X[i], 0 if y is None else y[i] , qid[i])
  return index

  
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
        avg_user: dictionary of an average user for mean imputation.
        avg_sim: dictionary of an average similarity relation.
        avg_conn: dictionary of an average connection strength relation.

      Returns:
        A triple with an list of features' lists, a list of true votes and a
      list with query ids. 
  """
  X = []
  y = []
  qid = []
  voter_map = {}
  q_count = 0
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
    qid.append(int(vote['voter']))
  return X, y, qid


def main():
  """ Predicts votes by applying LambdaMART technique.

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
   
    train_truth = [v['vote'] for v in train]
    if _BIAS:
      bias = BiasModel()
      train = bias.fit_transform(train, reviews)
 
    print 'Creating average user (for mean imputation)'
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

    print 'Outputting model'
    outfile = open('%s/rank_train-%s-%d.dat' % (_DATA_DIR, _CONF_STR, i), 'w')
    train_index = output_model(X_train, y_train, qid_train, outfile)
    outfile.close()
    outfile = open('%s/rank_val-%s-%d.dat' % (_DATA_DIR, _CONF_STR, i), 'w')
    val_index = output_model(X_val, None, qid_val, outfile)
    outfile.close()
    outfile = open('%s/rank_test-%s-%d.dat' % (_DATA_DIR, _CONF_STR, i), 'w')
    test_index = output_model(X_test, None, qid_test, outfile)
    outfile.close()

    for j in xrange(REP):
      print 'Fitting model'
      print getoutput(('java -jar lib/ranklib/RankLib.jar -train '
          '%s/rank_train-%s-%d.dat -save %s/lambdamart_model-%s-%d-%d.dat '
          '-gmax 5 -ranker 6 -metric2t NDCG@5 -tree %d -leaf %d -shrinkage '
          '%f') % (_DATA_DIR, _CONF_STR, i, _MODEL_DIR, _CONF_STR, i, j, _T,
          _L, _ALPHA)) 

      print 'Evaluating in train'
      print getoutput(('java -jar lib/ranklib/RankLib.jar -load '
          '%s/lambdamart_model-%s-%d-%d.dat -rank %s/rank_train-%s-%d.dat '
          '-score %s/rank_pred_train-%s-%d-%d.dat -gmax 5 -metric2T NDCG@5') % \
          (_MODEL_DIR, _CONF_STR, i, j, _DATA_DIR, _CONF_STR, i, _DATA_DIR,
          _CONF_STR, i, j))
      raw_pred = []
      predfile = open('%s/rank_pred_train-%s-%d-%d.dat' % (_DATA_DIR, _CONF_STR,
          i, j), 'r')
      raw_pred = [float(p.strip().split()[2]) for p in predfile]
      predfile.close()
      pred = [raw_pred[k] for k in train_index]
      if _BIAS:
        bias.add_bias(train, reviews, pred)
      print '~ Training error on set %d repetition %d' % (i, j)
      print 'RMSE: %f' % calculate_rmse(pred, train_truth)
      print 'nDCG@%d: %f' % (RANK_SIZE, calculate_avg_ndcg(train, reviews,
          pred, train_truth, RANK_SIZE))

      print 'Predicting in validation'
      print getoutput(('java -jar lib/ranklib/RankLib.jar -load '
          '%s/lambdamart_model-%s-%d-%d.dat -rank %s/rank_val-%s-%d.dat '
          '-score %s/rank_pred_val-%s-%d-%d.dat -gmax 5 -metric2T NDCG@5') % \
          (_MODEL_DIR, _CONF_STR, i, j, _DATA_DIR, _CONF_STR, i, _DATA_DIR,
          _CONF_STR, i, j))
      predfile = open('%s/rank_pred_val-%s-%d-%d.dat' % (_DATA_DIR, _CONF_STR,
          i, j), 'r')
      raw_pred = [float(p.strip().split()[2]) for p in predfile]
      predfile.close()
      pred = [raw_pred[k] for k in val_index]
      if _BIAS:
        bias.add_bias(val, reviews, pred)
      output = open('%s/lambdamart-%s-%d-%d.dat' % (_VAL_DIR, _CONF_STR, i, j),
          'w')
      for p in pred:
        print >> output, p
      output.close()
      
      print 'Predicting in test'
      print getoutput(('java -jar lib/ranklib/RankLib.jar -load '
          '%s/lambdamart_model-%s-%d-%d.dat -rank %s/rank_test-%s-%d.dat '
          '-score %s/rank_pred_test-%s-%d-%d.dat -gmax 5 -metric2T NDCG@5') % \
          (_MODEL_DIR, _CONF_STR, i, j, _DATA_DIR, _CONF_STR, i, _DATA_DIR,
          _CONF_STR, i, j))
      predfile = open('%s/rank_pred_test-%s-%d-%d.dat' % (_DATA_DIR, _CONF_STR,
          i, j), 'r')
      raw_pred = [float(p.strip().split()[2]) for p in predfile]
      predfile.close()
      pred = [raw_pred[k] for k in test_index]
      if _BIAS:
        bias.add_bias(test, reviews, pred)
      output = open('%s/lambdamart-%s-%d-%d.dat' % (_OUTPUT_DIR, _CONF_STR, i, 
          j), 'w')
      for p in pred:
        print >> output, p
      output.close()


if __name__ == '__main__':
  main()
