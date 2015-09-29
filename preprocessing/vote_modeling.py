""" Vote Modeling Module 
    --------------------

    Model votes and divides them between training, validation and test set
    chronologically.

    Usage:
      Used only as a module, not directly callable.
"""


from math import ceil
from datetime import datetime


def model_votes(reviews):
  """ Models votes with basic identification data (review id, reviewer id, voter 
      id), the vote value and the date of the review (used as an approximation for
      the vote date).

      Args:
        reviews: a dictionary with raw reviews.

      Returns:
        A list with dictionaries representing votes.
  """
  votes = []
  for review in reviews.values():
    for voter in review['votes']:
      vote = {}
      vote['review'] = review['id']
      vote['author'] = review['author']
      vote['voter'] = voter 
      vote['vote'] = review['votes'][voter]
      vote['date'] = review['date']
      vote['rel_vote'] = vote['vote'] - review['avg_vote'] 
          # there is a vote, so it cannot be nan
      votes.append(vote)
  return votes


def split_votes(votes):
  """ Splits votes between train and test sets. They are sorted chronologically
      (by review date) and the first half is used for train and the second,
      for test.

      Args:
        votes: list of modeled votes, represented as dictionaries.

      Returns:
        Three lists of vote dictionaries, the first representing the train set
      with 50% of votes, the second representing the validation set with 10% of
      votes, and the third is the test set with 40% of votes.
  """
  sorted_reviews = sorted(votes, key=lambda v:
      datetime.strptime(v['date'], '%d.%m.%Y'))
  first_cut_point = int(ceil(len(votes) / 2.0))
  second_cut_point = first_cut_point + int(ceil(len(votes) / 10.0))
  return votes[:first_cut_point], votes[first_cut_point:second_cut_point], \
      votes[second_cut_point:]
