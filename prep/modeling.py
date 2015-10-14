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


_NUM_THREADS = 7
#_SAMPLE = False 
#_SAMPLE_RATIO = 0.05 if _SAMPLE else 1
_OUTPUT_DIR = 'out/pkl'


def model():#ratio=_SAMPLE_RATIO):
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
  dump(trusts, open('%s/trusts.pkl' % _OUTPUT_DIR, 'w'))

 # if _SAMPLE:
 #   print 'Sampling reviews'
 #   reviews = {r['id']:r for r in sample_reviews(_SAMPLE_RATIO)}
 # else:
  reviews = {r['id']:r for r in parse_reviews()}

  print 'Modeling votes'
  votes = model_votes(reviews)
  sets = split_votes(votes)
  for (i, split) in enumerate(sets):
    train, val, test = split
    dump(train, open('%s/train-%d.pkl' % (_OUTPUT_DIR, i), 'w'))
    dump(val, open('%s/validation-%d.pkl' % (_OUTPUT_DIR, i), 'w'))
    dump(test, open('%s/test-%d.pkl' % (_OUTPUT_DIR, i), 'w'))
    print 'Modeling reviews %d' % i
    model_reviews_parallel(_NUM_THREADS, train, reviews)
    dump(reviews, open('%s/reviews-%d.pkl' % (_OUTPUT_DIR, i), 'w'))
    print 'Modeling users %d' % i
    test_users = set([v['author'] for v in val + test]) \
        .union(set([v['voter'] for v in val + test]))
    users = model_users(reviews, train, test_users, trusts)
    dump(users, open('%s/users-%d.pkl' % (_OUTPUT_DIR, i), 'w'))
    print 'Modeling author-voter interaction %d' % i
    test_pairs = [(v['author'], v['voter']) for v in val + test]
    sim = model_author_voter_similarity(train, users, test_pairs)
    dump(sim, open('%s/sim-%d.pkl' % (_OUTPUT_DIR, i), 'w'))
    conn = model_author_voter_connection(train, users, trusts, test_pairs)
    dump(conn, open('%s/conn-%d.pkl' % (_OUTPUT_DIR, i), 'w'))


if __name__ == '__main__':
  model()
