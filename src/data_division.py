""" Data division module 
    --------------------

    Divides the data (collection of votes) between training and test set
    chronologically.

    Usage:
      Used only as a module, not directly callable.
"""


from math import ceil
from datetime import datetime


""" Models votes with basic identification data (review id, reviewer id, rater
    id), the vote value and the date of the review (used as an approximation for
    the vote date).

    Args:
      reviews: a dictionary with raw reviews.

    Returns:
      A list with dictionaries representing votes.
"""
def model_votes(reviews):
  votes = []

  for review in reviews.values():
    for rater in review['votes']:
      vote = {}
      vote['review'] = review['id']
      vote['reviewer'] = review['user']
      vote['rater'] = rater
      vote['vote'] = review['votes'][rater]
      vote['date'] = review['date']
      votes.append(vote)

  return votes


""" Splits votes between train and test sets. They are sorted chronologically
    (by review date) and the first half is used for train and the second,
    for test.

    Args:
      reviews: dictionary of reviews to extract votes from. 

    Returns:
      Two lists of vote dictionaries, the first representing the train set and
    the second, the test set.
"""
def split_votes(reviews):
  votes = model_votes(reviews)
  sorted_reviews = sorted(votes, key=lambda v:
      datetime.strptime(v['date'], '%d.%m.%Y'))
  cut_point = int(ceil(len(votes) / 2.0))
  return votes[:cut_point], votes[cut_point:]
