""" Parser module
    -------------

    Parser raw information into memory under data structures formats.

    This module is used by other modules and should not be directly called.
"""

import nltk
from nltk.tokenize import wordpunct_tokenize, RegexpTokenizer, sent_tokenize
from nltk_contrib.readability.textanalyzer import syllables_en
from nltk.corpus import stopwords
from textblob import TextBlob
import networkx as nx


_FILENEW = 'data/rating_new.txt'
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
def parse_reviews(review_ids=None):
  f = open(_FILENEW, 'r')

  review_count = 0

  for l in f:
    l = l.strip().split('::::')
    review = {}
    review['id'] = review_count
    review_count += 1
    if review_ids and review['id'] not in review_ids:
      break
    review['user'] = l[0]
    review['product'] = l[1]
    review['category'] = l[2]
    review['rating'] = int(l[3]) / 10
    review['date'] = l[5]
    review['text'] = l[6]

    review['votes'] = parse_votes(l[7])

    yield review

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
  trust = nx.DiGraph()
  for l in f:
    l = l.strip().split('::::')
    trust.add_edge(l[0], l[1])
  f.close()
  return trust
