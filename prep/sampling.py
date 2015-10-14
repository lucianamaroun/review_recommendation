""" Sampler module
    --------------

    Sample reviews randomly.

    This module is used by other modules and should not be directly called.
"""


from random import sample
from math import ceil

from preprocessing.parsing import parse_reviews


def sample_reviews(sample_ratio):
  """ Samples reviews.

      Args:
        sample_ratio: the ratio amount of reviews that should be included in the
      sample.

      Returns:
        A list of sampled raw reviews.
  """
  reviews = [r for r in parse_reviews()]
  size = int(ceil(len(reviews) * sample_ratio))
  sel_reviews = sample(reviews, size)
  return sel_reviews
