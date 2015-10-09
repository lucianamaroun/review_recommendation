""" Vote Modeling Module 
    --------------------

    Model votes and divides them between training, validation and test set
    chronologically.

    Usage:
      Used only as a module, not directly callable.
"""


from math import ceil
from datetime import datetime

_SLIDE = 0.05
_SPLITS = 5
_TRAIN_RATIO = 0.4
_VAL_RATIO = 0.1

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
  votes = sorted(votes, key=lambda v: datetime.strptime(v['date'], '%d.%m.%Y'))
  sets = []
  size = len(votes)
  delta = int(ceil(size * _SLIDE))
  w_size = size - delta * (_SPLITS - 1)
  for i in xrange(_SPLITS):
    start = delta * i
    window = votes[start:start+w_size]
    train_cut = int(ceil(w_size * _TRAIN_RATIO))
    val_cut = train_cut + int(ceil(w_size * _VAL_RATIO))
    set_split = window[:train_cut], window[train_cut:val_cut], \
        window[val_cut:]
    sets.append(set_split)
  return sets 
