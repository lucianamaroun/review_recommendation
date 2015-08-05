from numpy import reshape, array


def map_review_features(review):
  """ Maps a review to array of features.

      Args:
        review: a review dictionary.

      Returns:
        A column array with features.
  """
  new_review = array([review['num_tokens'], review['num_sents'], 
      review['uni_ratio'], review['avg_sent'], review['cap_sent'],
      review['noun_ratio'], review['adj_ratio'], review['comp_ratio'],
      review['verb_ratio'], review['adv_ratio'], review['fw_ratio'],
      review['sym_ratio'], review['noun_ratio'], review['punct_ratio'],
      review['kl'], review['pos_ratio'], review['neg_ratio']])
  new_review = reshape(new_review, (17, 1))
  return new_review


def map_author_features(author):
  """ Maps a user to an array of author features.

      Args:
        author: a user dictionary of an author.

      Returns:
        A column array with features.
  """
  new_author = array([author['num_reviews'], author['avg_rating'],
      author['num_trustors'], author['num_trustees'], author['pagerank']])
  new_author = reshape(new_author, (5, 1))
  return new_author


def map_voter_features(voter):
  """ Maps a user to an array of voter features.

      Args:
        voter: a user dictionary of a voter.

      Returns:
        A column array with features.
  """
  new_voter = array([voter['num_trustors'], voter['num_trustees'],
      voter['pagerank'], voter['avg_rating'], voter['avg_rating_dir_net'],
      voter['avg_rating_sim'], voter['avg_help_giv'],
      voter['avg_help_giv_tru_net'], voter['avg_help_giv_sim']])
  new_voter = reshape(new_voter, (9, 1))
  return new_voter


def map_users_sim_features(users_sim):
  """ Maps a similary dictionary to an array of features.

      Args:
        users_sim: a dictionary containing similarity metrics between two users.

      Returns:
        A column array with features.
  """
  new_users_sim = array([users_sim['common_rated'], users_sim['jacc_rated'],
      users_sim['cos_ratings'], users_sim['pear_ratings'],
      users_sim['diff_avg_ratings'], users_sim['diff_max_ratings'],
      users_sim['diff_min_ratings']])
  new_users_sim = reshape(new_users_sim, (7, 1))
  return new_users_sim


def map_users_conn_features(users_conn):
  """ Maps a connection dictionary to an array of features.

      Args:
        users_sim: a dictionary containing connection metrics between two users.

      Returns:
        A column array with features.
  """
  new_users_conn = array([users_conn['jacc_trustees'],
      users_conn['jacc_trustors'], users_conn['adamic_adar_trustees'],
      users_conn['adamic_adar_trustors']#, users_conn['katz']
      ]) 
  new_users_conn = reshape(new_users_conn, (4, 1))
  return new_users_conn
