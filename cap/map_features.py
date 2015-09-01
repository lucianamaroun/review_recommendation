""" Map Features Module
    ------------------

    Map features from dictionaries of reviews, users or interactions.

    Usage: this method is not directly callable.
"""

from numpy import reshape, array, isnan


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
  model = array(model).reshape(17, 1)
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
  model = array(model).reshape(5, 1)
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
  model = array(model).reshape(9, 1)
  return model 


def map_users_sim_features(users_sim):
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
    model.append(users_sim[feature])
  model = array(model).reshape(7, 1)
  return model 


def map_users_conn_features(users_conn):
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
    model.append(users_conn[feature])
  model = array(model).reshape(5, 1)
  return model 


def compute_avg_user(users):
  """ Computes an average user from the set of users by averaging each features
      from users who have it defined.

      Args:
        users: dictionary of users dictionaries.

      Returns:
        A user dictionary containing averaged features, except identifiers and
      lists.
  """
  avg_user = users.itervalues().next().copy()
  for feature in avg_user:
    if feature in ['id', 'ratings', '_id']:
      continue
    avg_user[feature] = 0.0
    count = 0
    for user in users.itervalues():
      if not isnan(user[feature]):
        avg_user[feature] += user[feature]
        count += 1
    avg_user[feature] /= float(count)
  return avg_user

