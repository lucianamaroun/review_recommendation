""" Modeling module
    ---------------

    Models reviews, users and votes and generates train and test sets.

    Usage:
      $ python -m src.modeling
    on the project root directory.
"""

from src import sampler
from src import parser
from src.review_modeling import model_reviews_parallel
from src.user_modeling import model_users
from src.data_division import split_votes
from src.lib.sentiment.sentiwordnet import SimplifiedSentiWordNet


_NUM_THREADS = 8
_SAMPLE_RATIO = 0.5
_TRAIN_FILE = '/var/tmp/luciana/train%d-foreign.csv' % int(_SAMPLE_RATIO * 100)
_TEST_FILE = '/var/tmp/luciana/test%d-foreign.csv' % int(_SAMPLE_RATIO * 100)
_SAMPLE = True


""" Models reviews, users and votes in order to generate features from
    train and test sets. Aggregated statistics from test set includes reviews
    from both train and test set.

    Args:
      None.

    Returns:
      None. Calls output_model which outputs to files.
"""
def model():
  print 'Getting trust'
  trusts = parser.parse_trusts()

  print 'Modeling reviews'
  if _SAMPLE:
    sel_reviews = sampler.sample_reviews(_SAMPLE_RATIO)
    reviews = model_reviews_parallel(sel_reviews)
  else:
    reviews = model_reviews_parallel()

  print 'Split train and test'
  train, test = split_votes(reviews)

  print 'Modeling users'
  users = model_users(reviews, train, trusts)

  print 'Outputting'
  output_model(train, test, reviews, users, trusts)


""" Outputs feature model.

    Args:
      train: a list of votes belonging to train set.
      test: the list of votes belonging to test set.
      reviews: a dictionary of reviews.
      users: a dictionary of users with aggregated information from train
    set.
      trusts: a networkx DiGraph object.

    Returns:
      None. The output is inserted in _TRAIN_FILE and _TEST_FILE.
"""
def output_model(train, test, reviews, users, trusts):
  train_f = open(_TRAIN_FILE, 'w')
  test_f = open(_TEST_FILE, 'w')

  for out in [train_f, test_f]:
    print >> out, ('review_id,reviewer_id,rater_id,rating,'
        'num_chars,num_tokens,num_words,num_sents,unique_ratio,avg_sent,'
        'cap_ratio,noun_ratio,adj_ratio,adv_ratio,verb_ratio,comp_ratio,'
        'fw_ratio,sym_ratio,num_ratio,punct_ratio,'
        'pos_ratio,neg_ratio,kl_div,'
        'r_num_reviews,r_avg_rating,r_avg_help_rec,'
        'r_num_trustors,r_num_trustees,r_avg_help_giv,r_avg_rel_help_giv,'
        'r_sd_rating,r_sd_help_rec,r_sd_help_giv,r_pagerank,'
        'u_num_reviews,u_avg_rating,u_avg_help_rec,'
        'u_num_trustors,u_num_trustees,u_avg_help_giv,u_avg_rel_help_giv,'
        'u_sd_rating,u_sd_help_rec,u_sd_help_giv,u_pagerank,trust,truth')

  for partition, out in [(train, train_f), (test, test_f)]:
    for vote in partition:
      r = reviews[vote['review']]
      if r['user'] not in users or vote['rater'] not in users:
        continue
      rvr = users[r['user']]
      rtr = users[vote['rater']]
      trust = 1 if vote['rater'] in trusts and r['user'] in \
          trusts[vote['rater']] else 0
      print >> out, ('%s,%s,%s,%d,'
          '%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,' 
          '%d,%f,%f,%d,%d,%f,%f,%f,%f,%f,%f,'
          '%d,%f,%f,%d,%d,%f,%f,%f,%f,%f,%f,'
          '%d,%d') % (
          r['id'], r['user'], vote['rater'], r['rating'],
          r['num_chars'],r['num_tokens'],r['num_words'],r['num_sents'],
          r['uni_ratio'],r['avg_sent'],
          r['cap_sent'],
          r['noun_ratio'],r['adj_ratio'],r['adv_ratio'],r['verb_ratio'],
          r['comp_ratio'],r['fw_ratio'],r['sym_ratio'],r['num_ratio'],
          r['punct_ratio'],
          r['pos_ratio'],r['neg_ratio'],r['kl'],
          rvr['num_reviews'],rvr['avg_rating'],
          rvr['avg_help_rec'],rvr['num_trustors'],rvr['num_trustees'],
          rvr['avg_help_giv'],rvr['avg_rel_help_giv'],rvr['sd_rating'],
          rvr['sd_help_rec'],rvr['sd_help_giv'],rvr['pagerank'],
          rtr['num_reviews'],rtr['avg_rating'],
          rtr['avg_help_rec'],rtr['num_trustors'],rtr['num_trustees'],
          rtr['avg_help_giv'],rtr['avg_rel_help_giv'],rtr['sd_rating'],
          rtr['sd_help_rec'],rtr['sd_help_giv'],rtr['pagerank'],
          trust, vote['vote'])

  train_f.close()
  test_f.close()


if __name__ == '__main__':
  model()
