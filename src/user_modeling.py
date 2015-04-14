""" User modeling module
    --------------------

    Models users with a set of features derived from authoring and voting
    profile, as well as from the trust network. 

    Usage:
      Used only as a module, not directly callable.
"""


from numpy import std, mean
from networkx import pagerank
from scipy.spatial.distance import cosine

from src.author_voter_modeling import obtain_vectors # put in a specialized module

""" Initializes user features.

    Args:
      user_id: the id of the user to create the dictionary of.

    Returns:
      A dictionary with initialized values representing a user.
"""
def create_user(user_id):
  user = {}
  user['id'] = user_id
  if user_id == '5506501':
    print 'warm start'
    import sys
    sys.exit()
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
  return user


""" Creates a cold start user. 

    Args:
      user_id: the user id to create dictionary of.

    Returns:
      A dictionary with all features encoded as missing values. 
"""
def create_missing_user(user_id, trusts):
  prank = pagerank(trusts)
  user = {}
  user['id'] = user_id
  if user_id == '5506501':
    print 'cold start'
    import sys
    sys.exit()
  user['num_reviews'] = 0
  user['num_votes_rec'] = 0
  user['num_votes_giv'] = 0
  user['avg_rating'] = -1
  user['sd_rating'] = -1
  user['sd_help_rec'] = -1
  user['sd_help_giv'] = -1
  user['avg_rel_rating'] = -1
  user['avg_help_rec'] = -1
  user['avg_help_giv'] = -1
  user['avg_rel_help_giv'] = -1
  user['num_trustors'] = trusts.in_degree(user_id) if user_id in trusts else 0
  user['num_trustees'] = trusts.out_degree(user_id) if user_id in trusts else 0
  user['pagerank'] = prank[user_id] if user_id in prank else -1.0 
  user['avg_rating_sim'] = -1
  user['avg_help_giv_sim'] = -1
  if user_id not in trusts:
    user['avg_rating_dir_net'] = -1 
    user['avg_help_giv_tru_net'] = -1
  else:
    direct_net = trusts.predecessors(user_id) + trusts.successors(user_id)
    trust_net = trusts.successors(user_id)
    dir_net_avg = [users[n]['avg_rating'] for n in direct_net if n in users
        and users[n]['avg_rating'] >= 0]
    user['avg_rating_dir_net'] = mean(dir_net_avg) if dir_net_avg else -1
    tru_net_avg = [users[n]['avg_help_giv'] for n in trust_net if n in users
        and users[n]['avg_help_giv'] >= 0]
    user['avg_help_giv_tru_net'] = mean(tru_net_avg) if tru_net_avg else -1
    import numpy as np
    if np.isnan(user['avg_help_giv_tru_net']):
      print trust_net 
      print [users[s]['avg_help_giv'] for s in trust_net if s in users] 
      import sys
      sys.exit()
  return user


""" Adds user rating, updating related features.

    Args:
      user: the user dictionary whose fields must be updated.
      rating: the rating value associated to review.
      rel_rating: the relative rating of review.

    Returns:
      None. Changes are made in place.
"""
def add_user_rating(user, rating, rel_rating, product):
  user['num_reviews'] += 1
  user['avg_rating'] += float(rating)
  user['avg_rel_rating'] += float(rel_rating)
  user['ratings'][product] = float(rating)


""" Adds user vote, updating related features.

    Args:
      reviewer: the user dictionary of the reviewer.
      voter: the user dictionary of the voter.
      vote: the value of the vote.
      avg_help: the average helpfulness of the review being voted.

    Returns:
      None. Changes are made in place.
"""
def add_user_vote(reviewer, voter, vote, avg_help):
  reviewer['num_votes_rec'] += 1
  reviewer['avg_help_rec'] += int(vote)
  reviewer['sd_help_rec'].append(vote)

  voter['num_votes_giv'] += 1
  voter['avg_help_giv'] += int(vote)
  voter['avg_rel_help_giv'] +=  float(vote) - avg_help
  voter['sd_help_giv'].append(float(vote))


""" Finalizes user features, normalizing or aggregating features.

    Observation: -1 encodes missing values.

    Args:
      user: a user dictionary.

    Returns:
      None. Changes are made in place.
"""
def finalize_user_features(users, trusts):
  prank = pagerank(trusts)
  for user in users:
    if users[user]['num_reviews'] == 0:
      users[user]['avg_rating'] = -1 
      users[user]['avg_rel_rating'] = -1 
      users[user]['sd_rating'] = -1 
    else:
      users[user]['avg_rating'] /= float(users[user]['num_reviews'])
      users[user]['avg_rel_rating'] /= float(users[user]['num_reviews'])
      users[user]['sd_rating'] = std(users[user]['ratings'].values(), ddof=1)
    if users[user]['num_votes_rec'] == 0:
      users[user]['avg_help_rec'] = -1 
      users[user]['sd_help_rec'] = -1 
    else:
      users[user]['avg_help_rec'] /= float(users[user]['num_votes_rec'])
      users[user]['sd_help_rec'] = std(users[user]['sd_help_rec'], ddof=1)
    if users[user]['num_votes_giv'] == 0:
      users[user]['avg_help_giv'] = -1 
      users[user]['avg_rel_help_giv'] = -1 
      users[user]['sd_help_giv'] = -1 
    else:
      users[user]['avg_help_giv'] /= float(users[user]['num_votes_giv'])
      users[user]['avg_rel_help_giv'] /= float(users[user]['num_votes_giv'])
      users[user]['sd_help_giv'] = std(users[user]['sd_help_giv'], ddof=1)
    users[user]['pagerank'] = prank[user] if user in prank else -1.0
    import numpy as np
    if np.isnan(users[user]['avg_help_giv_tru_net']):
      print user
      import sys
      sys.exit()

""" Calculates aggregated features from immediate social network of user.

    Args:
      users: dictionary of users.
      trusts: nx.Digraph object with trust network.

    Returns:
      None. Changes are made in users dictionary.
"""
def calculate_network_agg_features(users, trusts):
  for u_id in users:
    if u_id not in trusts:
      users[u_id]['avg_rating_dir_net'] = -1 
      users[u_id]['avg_help_giv_tru_net'] = -1
      continue 
    direct_net = trusts.predecessors(u_id) + trusts.successors(u_id)
    trust_net = trusts.successors(u_id)
    dir_net_avg = [users[n]['avg_rating'] for n in direct_net if n in users
        and users[n]['avg_rating'] >= 0]
    users[u_id]['avg_rating_dir_net'] = mean(dir_net_avg) if dir_net_avg else -1
    tru_net_avg = [users[n]['avg_help_giv'] for n in trust_net if n in users
        and users[n]['avg_help_giv'] >= 0]
    users[u_id]['avg_help_giv_tru_net'] = mean(tru_net_avg) if tru_net_avg \
        else -1
    import numpy as np
    if np.isnan(users[u_id]['avg_help_giv_tru_net']):
      print trust_net
      print [users[s]['avg_help_giv'] for s in trust_net if s in users] 
      import sys
      sys.exit()


""" Calculates features related to statistics of user in the trust network.

    Args:
      users: dictionary of users.
      trusts: nx.Digraph object with trust network.

    Returns:
      None. Changes are made in users dictionary.
"""
def calculate_trust_features(users, trusts):
  for user in users:
    if user not in trusts:
      users[user]['num_trustors'] = 0 
      users[user]['num_trustees'] = 0 
      continue
    users[user]['num_trustors'] = trusts.in_degree(user)
    users[user]['num_trustees'] = trusts.out_degree(user)


""" Groups votes by review.

    Args:
      votes: a list of votes' dictionaries.

    Returns:
      A dictionary indexed by review ids and containing a list of votes as
    values.
"""
def group_votes_by_review(votes):
  grouped_votes = {}

  for vote in votes:
    if vote['review'] not in grouped_votes:
      grouped_votes[vote['review']] = []
    grouped_votes[vote['review']].append(vote)

  return grouped_votes


""" Calculates aggregated features related to similar users.
    
    Args:
      users: dictionary of users.
      similar: dictionary of similar users.

    Returns:
      None. Changes are made in users dictionary.
"""
def calculate_similar_agg_features(users, similar):
  for user in similar:
    sim_avg = [users[s]['avg_rating'] for s in similar[user]
        if users[s]['avg_rating'] >= 0]
    users[user]['avg_rating_sim'] = mean(sim_avg) if sim_avg else -1
    sim_avg = [users[s]['avg_help_giv'] for s in similar[user]
        if users[s]['avg_help_giv'] >= 0]
    users[user]['avg_help_giv_sim'] = mean(sim_avg) if sim_avg else -1
    import numpy as np
    if np.isnan(users[user]['avg_help_giv_sim']):
      print similar[user]
      print [users[s]['avg_help_giv'] for s in similar[user] if s in users] 
      import sys
      sys.exit()

""" Gets similar users for each user. A user B is amongst user A similar users
    if their cosine rating similarity is higher than the average similarity of
    A with all the users.

    Args:
      users: a dictionary of users, containing user ids as keys and user
        dictionaries as values.

    Returns:
      A dictionary of similar users indexed by user ids and having a list of
    similar user ids as value.
"""
def get_similar_users(users):
  sim = {}
  for user in users:
    sim[user] = {}
  for user_a in users.keys():
    a_ratings = users[user_a]['ratings']
    for user_b in [u for u in users.keys() if u > user_a]:
      b_ratings = users[user_b]['ratings']
      vec_a, vec_b = obtain_vectors(a_ratings, b_ratings)
      sim[user_a][user_b] = sim[user_b][user_a] = 1 - cosine(vec_a, vec_b)
  sim_users = {}
  for user in sim:
    avg = mean(sim[user].values())
    sim_users[user] = [u for u in sim[user] if sim[user][u] > avg]
  return sim_users
  

""" Models users, aggregating information from reviews and trust relations.

    Args:
      reviews: a dictionary with modeled reviews.
      train: a list of votes used as train.
      trusts: a networkx DiGraph object.

    Returns:
      A dictionary of users indexed by user ids.
"""
def model_users(reviews, train, trusts):
  users = {}

  grouped_train = group_votes_by_review(train)
  for review_id in grouped_train:
    review = reviews[review_id]
    if review['user'] not in users:
      rev_dict = create_user(review['user'])
      users[review['user']] = rev_dict
    add_user_rating(users[review['user']], review['rating'],
        review['rel_rating'], review['product'])
    avg_help = float(sum([v['vote'] for v in grouped_train[review_id]])) / \
        len(grouped_train[review_id])
    for vote in grouped_train[review_id]:
      if vote['voter'] not in users:
        rat_dict = create_user(vote['voter'])
        users[vote['voter']] = rat_dict
      add_user_vote(users[review['user']], users[vote['voter']], vote['vote'],
          avg_help)
  calculate_trust_features(users, trusts)
  finalize_user_features(users, trusts)
  calculate_network_agg_features(users, trusts)
  similar = get_similar_users(users)
  calculate_similar_agg_features(users, similar)

  return users

