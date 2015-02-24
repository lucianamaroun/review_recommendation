""" User modeling module
    --------------------

    Models users with a set of features derived from authoring and voting
    profile, as well as from the trust network. 

    Usage:
      Used only as a module, not directly callable.
"""


from numpy import std
from networkx import pagerank


""" Initializes user features.

    Args:
      user_id: the id of the user to create the dictionary of.

    Returns:
      A dictionary with initialized values representing a user.
"""
def create_user(user_id):
  user = {}
  user['id'] = user_id
  user['num_reviews'] = 0
  user['num_votes_rec'] = 0
  user['num_votes_giv'] = 0
  user['avg_rating'] = 0
  user['sd_rating'] = []
  user['sd_help_rec'] = []
  user['sd_help_giv'] = []
  user['avg_rel_rating'] = 0
  user['avg_help_rec'] = 0
  user['avg_help_giv'] = 0
  user['avg_rel_help_giv'] = 0
  user['num_trustees'] = 0
  user['num_trustors'] = 0
  return user


""" Adds user rating, updating related features.

    Args:
      user: the user dictionary whose fields must be updated.
      rating: the rating value associated to review.
      rel_rating: the relative rating of review.

    Returns:
      None. Changes are made in place.
"""
def add_user_rating(user, rating, rel_rating):
  user['num_reviews'] += 1
  user['avg_rating'] += float(rating)
  user['avg_rel_rating'] += float(rel_rating)
  user['sd_rating'].append(float(rating))


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

    Args:
      user: a user dictionary.

    Returns:
      None. Changes are made in place.
"""
def finalize_user_features(users, trusts):
  prank = pagerank(trusts)
  remove_users = set()
  for user in users:
    if users[user]['num_reviews'] == 0 or users[user]['num_votes_rec'] == 0 \
        or users[user]['num_votes_giv'] == 0:
      remove_users.add(user) # removing cold starts
      continue
    users[user]['avg_rating'] /= float(users[user]['num_reviews'])
    users[user]['avg_rel_rating'] /= float(users[user]['num_reviews'])
    users[user]['sd_rating'] = std(users[user]['sd_rating'], ddof=1)
    users[user]['avg_help_rec'] /= float(users[user]['num_votes_rec'])
    users[user]['sd_help_rec'] = std(users[user]['sd_help_rec'], ddof=1)
    users[user]['avg_help_giv'] /= float(users[user]['num_votes_giv'])
    users[user]['avg_rel_help_giv'] /= float(users[user]['num_votes_giv'])
    users[user]['sd_help_giv'] = std(users[user]['sd_help_giv'], ddof=1)
    users[user]['pagerank'] = prank[user] if user in prank else 0.0
  for user in remove_users:
    users.pop(user, None)


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
        review['rel_rating'])
    avg_help = float(sum([v['vote'] for v in grouped_train[review_id]])) / \
        len(grouped_train[review_id])
    for vote in grouped_train[review_id]:
      if vote['voter'] not in users:
        rat_dict = create_user(vote['voter'])
        users[vote['voter']] = rat_dict
      add_user_vote(users[review['user']], users[vote['voter']], vote['vote'],
          avg_help)
  for user in trusts:
    users[user]['num_trustors'] = trusts.in_degree(user)
    users[user]['num_trustees'] = trusts.out_degree(user)
  finalize_user_features(users, trusts)

  return users
