""" Sampler module
    --------------

    Sample reviews randomly.

    This module is used by other modules and should not be directly called.
"""

import random
import math

from src import parser


""" Samples reviews.

    Args:
      sample_ratio: the ratio amount of reviews that should be included in the
    sample.

    Returns:
      A list of sampled raw reviews.
"""
def sample(sample_ratio):
  reviews = [r for r in parser.parse_reviews()]

  sample_reviews = random.sample(reviews, int(math.ceil(len(reviews) *
      sample_ratio)))

  return sample_reviews
