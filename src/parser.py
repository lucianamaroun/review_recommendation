""" Parser module
    -------------

    Parser raw information into memory under data structures formats.

    This module is used by other modules and should not be directly called.
"""

from networkx import DiGraph


_FILENEW = 'data/rating.txt'
PUNCTUATION = [';', ':', ',', '.', '!', '?']
    # source: http://www.nltk.org/api/nltk.tokenize.html


""" Iterates through reviews, parsing the file content.

    Args:
        None.

    Returns:
        A dictionary representing the review with keys: "id",
    "user", "product", "category", "rating", "rel_rating",
    "date", "text", "votes" (dictionary
    indexed by rater ids with helfulness votes as values).
"""
def parse_reviews(verbose=False):
  f = open(_FILENEW, 'r')

  review_count = 0

  count_lines = 0
  count_ignored = 0
  type_ignored = {'user': 0, 'product': 0, 'category': 0, 'rating': 0,
      'date': 0, 'text': 0, 'votes': 0}
  for l in f:
    count_lines += 1

    try:
      l = l.strip().split('::::')
      review = {}
      review['id'] = review_count
      review_count += 1

      if l[0].strip():
        review['user'] = l[0].strip()
      else:
        raise Exception('user')
      if l[1].strip():
        review['product'] = l[1].strip()
      else:
        raise Exception('product')
      if l[2].strip():
        review['category'] = l[2].strip()
      else:
        raise Exception('category')
      try:
        review['rating'] = int(l[3]) / 10
      except Exception:
        raise Exception('rating')
      if l[5].strip():
        review['date'] = l[5].strip()
      else:
        raise Exception('date')
      if l[6].strip():
        review['text'] = l[6].strip()
      else:
        raise Exception('text')
      try:
        review['votes'] = parse_votes(l[7])
      except Exception:
        raise Exception('votes')
    except Exception as e:
      if verbose:
        print 'Exception on parsing review, line %d, type %s' % (count_lines,
            e.args[0])
        print l
        print '--------------------------'
        count_ignored += 1
        if e.args[0] in type_ignored:
          type_ignored[e.args[0]] += 1
      continue

    yield review

  if verbose:
    print '#############################'
    print 'Summary of Parsing Errors:'
    print '~ Ignored: %d' % count_ignored
    for item in type_ignored.items():
      print '~ Ignored of type %s: %d' % item
    print '#############################'

  f.close()


""" Parses review votes from raw string.

    Args:
        raw_votes: the string containing the raw votes from input file.

    Returns:
        A dictionary from rater, represented by an user id, to vote, represented by and integer f
    from 0 to 5. sorted_votes consists in a list of tuples (user, vote) sorted by voting time.
"""
def parse_votes(raw_votes):
    votes = {}
    str_votes = raw_votes.split(':::')

    for vote in str_votes:
      if vote == '</endperson>':
        break
      user, help_vote = vote.split(':')
      help_vote = int(help_vote)
      if user not in votes:
        # avoid duplication: seems that when there is a also a comment, the vote is duplicated
        votes[user] = help_vote

    return votes


""" Parses basic statistics from the users but, instead of returning
    all together, each user is yield at a time.

    Args:
        None.

    Yields:
        A dictionary representing an user, containing keys "id",
    "since", "count", and "trustors".
"""
def parses_userstat():
  f = open('data/userstatistic.txt', 'r')
  for l in f:
    l = l.strip().split('::::')
    user = {}
    user['id'] = l[0]
    user['since'] = l[1]
    user['count'] = int(l[2])
    user['trustors'] = int(l[3])
    yield user
  f.close()


""" Parses trust relations.

    Args:
        None.

    Returns:
        A dictionary with user ids as indexes and a list of user ids as values.
    This means that the user represented by the index trusts the corresponding
    list of users.
"""
def parse_trusts():
  f = open('data/trustnetwork.txt', 'r')
  trust = DiGraph()
  for l in f:
    l = l.strip().split('::::')
    trust.add_edge(l[0], l[1])
  f.close()
  return trust
