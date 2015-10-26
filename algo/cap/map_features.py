""" Map Features Module
    ------------------

    Maps features from dictionaries of reviews, users or interactions.

    Usage: this method is not directly callable.
"""


from numpy import reshape, array, isnan

from util.avg_model import compute_avg_user, compute_avg_model


def map_review_features(review):
  """ Maps a review to array of features.

      Args:
        review: a review dictionary.

      Returns:
        A column array with features.
  """
  features = ['num_tokens', 'num_sents', 'uni_ratio', 'avg_sent', 'cap_sent',
      'noun_ratio', 'adj_ratio', 'comp_ratio', 'verb_ratio', 'adv_ratio',
      'fw_ratio', 'sym_ratio', 'noun_ratio', 'punct_ratio', 'kl', 'pos_ratio',
      'neg_ratio']
  model = []
  for feature in features:
    model.append(review[feature])
  model = array(model)
  return model 


def map_author_features(author, avg_user):
  """ Maps a user to an array of author features.

      Args:
        author: a user dictionary of an author.
        avg_user: a dictionary of an average user for mean imputation of
          undefined features.

      Returns:
        A column array with features.
  """
  features = ['num_reviews', 'avg_rating', 'num_trustors', 'num_trustees',
      'pagerank']
  model = []
  for feature in features:
    model.append(avg_user[feature] if isnan(author[feature]) else 
        author[feature])
  model = array(model)
  return model 


def map_voter_features(voter, avg_user):
  """ Maps a user to an array of voter features.

      Args:
        voter: a user dictionary of a voter.
        avg_user: a dictionary of an average user for mean imputation of
          undefined features.

      Returns:
        A column array with features.
  """
  features = ['num_trustors', 'num_trustees', 'pagerank', 'avg_rating', 
      'avg_rating_dir_net', 'avg_rating_sim', 'avg_help_giv', 
      'avg_help_giv_tru_net', 'avg_help_giv_sim'] 
  model = []
  for feature in features:
    model.append(avg_user[feature] if isnan(voter[feature]) else voter[feature])
  model = array(model)
  return model 


def map_users_sim_features(users_sim, avg_sim):
  """ Maps a similary dictionary to an array of features.

      Args:
        users_sim: a dictionary containing similarity metrics between two users.

      Returns:
        A column array with features.
  """
  features = ['common_rated', 'jacc_rated', 'cos_ratings', 'pear_ratings',
      'diff_avg_ratings', 'diff_max_ratings', 'diff_min_ratings']
  model = []
  for feature in features:
    if isnan(users_sim[feature]):
      model.append(avg_sim[feature])
      print ' -- Imputed on sim'
    else:
      model.append(users_sim[feature])
  model = array(model)
  return model 


def map_users_conn_features(users_conn, avg_conn):
  """ Maps a connection dictionary to an array of features.

      Args:
        users_sim: a dictionary containing connection metrics between two users.

      Returns:
        A column array with features.
  """
  features = ['jacc_trustees', 'jacc_trustors', 'adamic_adar_trustees',
      'adamic_adar_trustors', 'katz'] 
  model = []
  for feature in features:
    if isnan(users_conn[feature]):
      model.append(avg_conn[feature])
      print ' -- Imputed on conn'
    else:
      model.append(users_conn[feature])
  model = array(model)
  return model 


def map_features(votes, reviews, users, users_sim, users_conn, trusts):
  avg_user = compute_avg_user(users)
  avg_sim = compute_avg_model(users_sim)
  avg_conn = compute_avg_model(users_conn)
  features = {'review': [], 'author': [], 'voter': [], 'sim': [], 'conn': []}
  for vote in votes:
    r_id, a_id, v_id = vote['review'], vote['author'], vote['voter']
    r_feat = map_review_features(reviews[r_id])
    features['review'].append(r_feat)
    author = users[a_id] if a_id in users else avg_user
    a_feat = map_author_features(author, avg_user)
    features['author'].append(a_feat)
    voter = users[v_id] if v_id in users else avg_user
    v_feat = map_voter_features(voter, avg_user)
    features['voter'].append(v_feat)
    if v_id in users and a_id in users[v_id]['similars']:
      if (a_id, v_id) in users_sim:
        sim = users_sim[(a_id, v_id)]
        sim_feat = map_users_sim_features(sim, avg_sim)
        features['sim'].append(sim_feat)
     # sim = users_sim[(a_id, v_id)] if (a_id, v_id) in users_sim else avg_sim
    if v_id in trusts and a_id in trusts[v_id]:
     # conn = users_conn[(a_id, v_id)] if (a_id, v_id) in users_conn else \
     #     avg_conn
      if (a_id, v_id) in users_conn:
        conn = users_conn[(a_id, v_id)]
        conn_feat = map_users_conn_features(conn, avg_conn)
        features['conn'].append(conn_feat)
  return features
