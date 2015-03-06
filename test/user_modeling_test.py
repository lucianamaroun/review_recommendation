import unittest

import src.user_modeling as umod


class SimilarTestCase(unittest.TestCase):
  """ Test case of similar users calculation and derived features. """
  def setUp(self):
    self.users = {
      1: {'id': 1, 'ratings': {1: 1, 2: 2, 4: 5, 5: 2}, 'avg_rating': 2.5, 
        'avg_help_giv': 4.8},
      2: {'id': 2, 'ratings': {2: 4, 5: 2, 6: 0}, 'avg_rating': 2.0,
        'avg_help_giv': 4.2},
      3: {'id': 3, 'ratings': {2: 5, 4: 5, 5: 4}, 'avg_rating': 14.0/3,
        'avg_help_giv': 3.8},
      4: {'id': 4, 'ratings': {1: 3, 3: 4, 6: 5}, 'avg_rating': 4.0,
        'avg_help_giv': 4.0},
      5: {'id': 5, 'ratings': {3: 2, 4: 4, 5: 4, 6: 3}, 'avg_rating': 13.0/4,
        'avg_help_giv': 4.5}
    }
    self.similar = {1: [3, 5], 2: [1, 3], 3: [1, 2, 5], 4: [5], 5: [1, 3]}
    self.new_users = {
      1: {'id': 1, 'ratings': {1: 1, 2: 2, 4: 5, 5: 2}, 'avg_rating': 2.5,
        'avg_help_giv': 4.8, 'avg_rating_sim': (14.0/3 + 13.0/4)/2,
        'avg_help_giv_sim': (3.8 + 4.5) / 2},
      2: {'id': 2, 'ratings': {2: 4, 5: 2, 6: 0}, 'avg_rating': 2.0,
        'avg_help_giv': 4.2, 'avg_rating_sim': (2.5 + 14.0/3)/2,
        'avg_help_giv_sim': (4.8 + 3.8)/2},
      3: {'id': 3, 'ratings': {2: 5, 4: 5, 5: 4}, 'avg_rating': 14.0/3,
        'avg_help_giv': 3.8, 'avg_rating_sim': (2.5 + 2 + 13.0/4)/3,
        'avg_help_giv_sim': (4.8 + 4.2 + 4.5)/3},
      4: {'id': 4, 'ratings': {1: 3, 3: 4, 6: 5}, 'avg_rating': 4.0,
        'avg_help_giv': 4.0, 'avg_rating_sim': 13.0/4, 'avg_help_giv_sim': 4.5},
      5: {'id': 5, 'ratings': {3: 2, 4: 4, 5: 4, 6: 3}, 'avg_rating': 13.0/4,
        'avg_help_giv': 4.5, 'avg_rating_sim': (2.5 + 14.0/3)/2,
        'avg_help_giv_sim': (4.8 + 3.8)/2}
    }

  def test_get_similar_users(self):
    similar = umod.get_similar_users(self.users)
    self.assertDictEqual(similar, self.similar)

  def test_calculate_similar_agg_features(self):
    umod.calculate_similar_agg_features(self.users, self.similar)
    for u in self.users:
      self.users[u]['avg_rating_sim'] = round(self.users[u]['avg_rating_sim'], 
          16)
      self.users[u]['avg_help_giv_sim'] = \
          round(self.users[u]['avg_help_giv_sim'], 16)
    self.assertDictEqual(self.users, self.new_users)


if __name__ == '__main__':
  unittest.main()
