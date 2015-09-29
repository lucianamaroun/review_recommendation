""" User modeling module
    --------------------

    Models users with a set of features derived from authoring and voting
    profile, as well as from the trust network. 

    Usage:
      Used only as a module, not directly callable.
"""


from numpy import std, mean, nan, isnan
from networkx import pagerank

from util.aux import cosine, vectorize


def create_user(user_id):
  """ Initializes user features.

      Args:
        user_id: the id of the user to create the dictionary of.

      Returns:
        A dictionary with initialized values representing a user.
  """
  user = {}
  user['id'] = user_id
  user['num_reviews'] = 0
  user['num_votes_rec'] = 0
  user['num_votes_giv'] = 0
  user['ratings'] = {} 
  user['avg_rating'] = 0
  user['sd_rating'] = 0 
  user['sd_help_rec'] = []
  user['sd_help_giv'] = []
  user['avg_rel_rating'] = 0
  user['avg_help_rec'] = 0
  user['avg_help_giv'] = 0
  user['avg_rel_help_giv'] = 0
  user['num_trustees'] = 0
  user['num_trustors'] = 0
  user['pagerank'] = 0
  user['avg_rating_sim'] = 0
  user['avg_help_giv_sim'] = 0
  user['avg_rating_dir_net'] = 0 
  user['avg_help_giv_tru_net'] = 0
  user['similars'] = set()
  return user


def add_user_rating(user, rating, rel_rating, product):
  """ Adds user rating, updating related features.

      Args:
        user: a dictionary of the users whose fields must be updated.
        rating: an integer with the rating value associated to review.
        rel_rating: a float with the relative rating of review.
        product: a string with the name of the product.

      Returns:
        None. Changes are made in place.
  """
  user['num_reviews'] += 1
  user['avg_rating'] += float(rating)
  user['avg_rel_rating'] += float(rel_rating)
  user['ratings'][product] = float(rating)


def add_user_vote(reviewer, voter, vote, avg_help):
  """ Adds user vote, updating related features.

      Args:
        reviewer: a dictionary representing the reviewer.
        voter: a dictionary representing the voter.
        vote: an integer with the value of the vote.
        avg_help: a float with the average helpfulness of the voted review.

      Returns:
        None. Changes are made in place.
  """
  reviewer['num_votes_rec'] += 1
  reviewer['avg_help_rec'] += int(vote)
  reviewer['sd_help_rec'].append(vote)

  voter['num_votes_giv'] += 1
  voter['avg_help_giv'] += int(vote)
  voter['avg_rel_help_giv'] +=  float(vote) - avg_help
  voter['sd_help_giv'].append(float(vote))


def finalize_vote_related_features(users):
  """ Finalizes user features, normalizing or aggregating features.

      Observation:
      - nan encodes missing values.

      Args:
        users: dictionary of user dictionaries.

      Returns:
        None. Changes are made in place.
  """
  for user in users:
    if users[user]['num_reviews'] == 0:
      users[user]['avg_rating'] = nan 
      users[user]['avg_rel_rating'] = nan 
      users[user]['sd_rating'] = nan
    else:
      users[user]['avg_rating'] /= float(users[user]['num_reviews'])
      users[user]['avg_rel_rating'] /= float(users[user]['num_reviews'])
      users[user]['sd_rating'] = std(users[user]['ratings'].values(), ddof=1) \
          if users[user]['num_reviews'] > 1 else 0.0
    if users[user]['num_votes_rec'] == 0:
      users[user]['avg_help_rec'] = nan 
      users[user]['sd_help_rec'] = nan
    else:
      users[user]['avg_help_rec'] /= float(users[user]['num_votes_rec'])
      users[user]['sd_help_rec'] = std(users[user]['sd_help_rec'], ddof=1) \
          if users[user]['num_votes_rec'] > 1 else 0.0
    if users[user]['num_votes_giv'] == 0:
      users[user]['avg_help_giv'] = nan 
      users[user]['avg_rel_help_giv'] = nan 
      users[user]['sd_help_giv'] = nan
    else:
      users[user]['avg_help_giv'] /= float(users[user]['num_votes_giv'])
      users[user]['avg_rel_help_giv'] /= float(users[user]['num_votes_giv'])
      users[user]['sd_help_giv'] = std(users[user]['sd_help_giv'], ddof=1) \
          if users[user]['num_votes_giv'] > 1 else 0.0


def calculate_network_agg_features(users, trusts):
  """ Calculates aggregated features from immediate social network of user.

      Observation:
      - nan encodes missing values.

      Args:
        users: dictionary of users.
        trusts: nx.Digraph object with trust network.

      Returns:
        None. Changes are made in users dictionary.
  """
  prank = pagerank(trusts)
  for u_id in users:
    users[u_id]['pagerank'] = prank[u_id] if u_id in prank else nan
    if u_id not in trusts:
      users[u_id]['avg_rating_dir_net'] = nan 
      users[u_id]['avg_help_giv_tru_net'] = nan
      continue 
    direct_net = trusts.predecessors(u_id) + trusts.successors(u_id)
    trust_net = trusts.successors(u_id)
    dir_net_avg = [users[n]['avg_rating'] for n in direct_net if n in users
        and not isnan(users[n]['avg_rating'])]
    users[u_id]['avg_rating_dir_net'] = mean(dir_net_avg) if dir_net_avg else nan 
    tru_net_avg = [users[n]['avg_help_giv'] for n in trust_net if n in users
        and not isnan(users[n]['avg_help_giv'])]
    users[u_id]['avg_help_giv_tru_net'] = mean(tru_net_avg) if tru_net_avg \
        else nan


def calculate_trust_features(users, test_users, trusts):
  """ Calculates features related to statistics of user in the trust network.

      Args:
        users: dictionary of users.
        test_users: ids of users which are in test.
        trusts: nx.Digraph object with trust network.

      Returns:
        None. Changes are made in users dictionary.
  """
  for user in trusts: 
    if user not in users and user in test_users:
      # cold-start in test but in trust network
      users[user] = create_user(user)
      users[user]['sd_help_rec'] = nan
      users[user]['sd_help_giv'] = nan
  for user in users:
    if user not in trusts:
      users[user]['num_trustors'] = 0 
      users[user]['num_trustees'] = 0 
    else:
      users[user]['num_trustors'] = trusts.in_degree(user)
      users[user]['num_trustees'] = trusts.out_degree(user)


def group_votes_by_review(votes):
  """ Groups votes by review.

      Args:
        votes: a list of votes' dictionaries.

      Returns:
        A dictionary indexed by review ids and containing a list of votes as
      values.
  """
  grouped_votes = {}

  for vote in votes:
    if vote['review'] not in grouped_votes:
      grouped_votes[vote['review']] = []
    grouped_votes[vote['review']].append(vote)

  return grouped_votes


def calculate_similar_agg_features(users):
  """ Calculates aggregated features related to similar users.
      
      Args:
        users: dictionary of users.

      Returns:
        None. Changes are made in users dictionary.
  """
  for user in users:
    sim_avg = [users[s]['avg_rating'] for s in users[user]['similars']
        if not isnan(users[s]['avg_rating'])]
    users[user]['avg_rating_sim'] = mean(sim_avg) if sim_avg else nan 
    sim_avg = [users[s]['avg_help_giv'] for s in users[user]['similars']
        if not isnan(users[s]['avg_help_giv'])]
    users[user]['avg_help_giv_sim'] = mean(sim_avg) if sim_avg else nan 


def calculate_similar_users(users):
  """ Gets similar users for each user. A user B is amongst user A similar users
      if their cosine rating similarity is higher than the average similarity of
      A with all the users.

      Args:
        users: a dictionary of users, containing user ids as keys and user
          dictionaries as values.

      Returns:
        None. Changes are made in place by adding a 'similars' key in each user
      dictionary with a list of similar users' ids.
  """
  sim = {}
  for user in users:
    sim[user] = {}
  for user_a in users:
    a_ratings = users[user_a]['ratings']
    for user_b in [u for u in users if u > user_a]:
      b_ratings = users[user_b]['ratings']
      vec_a, vec_b = vectorize(a_ratings, b_ratings)
      sim[user_a][user_b] = sim[user_b][user_a] = cosine(vec_a, vec_b)
  for user in users:
    avg = mean(sim[user].values())
    users[user]['similars'] = set([u for u in sim[user] if sim[user][u] >= avg])


def model_users(reviews, train, test_users, trusts):
  """ Models users, aggregating information from reviews and trust relations.

      Args:
        reviews: a dictionary with modeled reviews.
        train: a list of votes used as train.
        test_users: a list of ids of users which are in test set.
        trusts: a networkx DiGraph object.

      Returns:
        A dictionary of users indexed by user ids.
  """
  users = {}

  grouped_train = group_votes_by_review(train)
  for review_id in grouped_train:
    review = reviews[review_id]
    if review['author'] not in users:
      rev_dict = create_user(review['author'])
      users[review['author']] = rev_dict
    add_user_rating(users[review['author']], review['rating'],
        review['rel_rating'], review['product'])
    avg_help = float(sum([v['vote'] for v in grouped_train[review_id]])) / \
        len(grouped_train[review_id])
    for vote in grouped_train[review_id]:
      if vote['voter'] not in users:
        rat_dict = create_user(vote['voter'])
        users[vote['voter']] = rat_dict
      add_user_vote(users[review['author']], users[vote['voter']], vote['vote'],
          avg_help)
  finalize_vote_related_features(users)
  calculate_trust_features(users, test_users, trusts)
  calculate_network_agg_features(users, trusts)
  calculate_similar_users(users)
  calculate_similar_agg_features(users)

  return users

