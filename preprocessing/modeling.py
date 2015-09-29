""" Modeling module
    ---------------

    Models reviews, users, votes and users interactions. 

    Usage:
      $ python -m preprocessing.model
    on the project root directory.
"""


from pickle import load, dump

from preprocessing.sampling import sample_reviews
from preprocessing.parsing import parse_trusts, parse_reviews
from preprocessing.review_modeling import model_reviews_parallel
from preprocessing.user_modeling import model_users
from preprocessing.vote_modeling import model_votes, split_votes
from preprocessing.interaction_modeling import model_author_voter_similarity, \
    model_author_voter_connection


_NUM_THREADS = 6
_SAMPLE = True 
_SAMPLE_RATIO = 0.001
_OUTPUT_DIR = 'out/pkl'


def model(ratio=_SAMPLE_RATIO):
  """ Models reviews, users and votes in order to generate features from
      train and test sets. Aggregated statistics from test set includes reviews
      from both train and test set.

      Args:
        ratio (optional): a float in range [0, 1] with the rate of reviews to
      sample.

      Returns:
        None. Modeled structures are serialized into pickle files. 
  """
  print 'Getting trust'
  trusts = parse_trusts()
  dump(trusts, open('%s/trusts%.2f.pkl' % (_OUTPUT_DIR, _SAMPLE_RATIO * 100), 'w'))

  if _SAMPLE:
    print 'Sampling reviews'
    reviews = {r['id']:r for r in sample_reviews(_SAMPLE_RATIO)}
  else:
    reviews = {r['id']:r for r in parse_reviews()}

  print 'Modeling votes'
  votes = model_votes(reviews)
  train, validation, test = split_votes(votes)
  dump(train, open('%s/train%.2f.pkl' % (_OUTPUT_DIR, _SAMPLE_RATIO * 100), 'w'))
  dump(validation, open('%s/validation%.2f.pkl' % (_OUTPUT_DIR, _SAMPLE_RATIO * 100), 'w'))
  dump(test, open('%s/test%.2f.pkl' % (_OUTPUT_DIR, _SAMPLE_RATIO * 100), 'w'))
    
  print 'Modeling reviews'
  model_reviews_parallel(_NUM_THREADS, train, reviews)
  dump(reviews, open('%s/reviews%.2f.pkl' % (_OUTPUT_DIR, _SAMPLE_RATIO * 100), 'w'))

  print 'Modeling users'
  test_users = set([v['author'] for v in test]).union(set([v['voter'] for v in
      test]))
  users = model_users(reviews, train, test_users, trusts)
  dump(users, open('%s/users%.2f.pkl' % (_OUTPUT_DIR, _SAMPLE_RATIO * 100), 'w'))

  print 'Modeling author-voter interaction'
  test_pairs = [(v['author'], v['voter']) for v in test]
  sim = model_author_voter_similarity(train, users, test_pairs)
  dump(sim, open('%s/sim%.2f.pkl' % (_OUTPUT_DIR, _SAMPLE_RATIO * 100), 'w'))
  conn = model_author_voter_connection(train, users, trusts, test_pairs)
  dump(conn, open('%s/conn%.2f.pkl' % (_OUTPUT_DIR, _SAMPLE_RATIO * 100), 'w'))


if __name__ == '__main__':
  model()
