""" Review sd script
    ----------------

    Generates and output reviews' standard deviation of votes.

    Usage:
      $ python -m script.review_sd
"""


from numpy import std

from src.parser import parse_reviews


_OUTPUT_R = open('out/review_sd.dat', 'w')
_OUTPUT_U = open('out/user_sd.dat', 'w')


for review in parse_reviews():
  print >> _OUTPUT_R, '%d,%f' % (review['id'], std(review['votes'].values()))

users = {}
for review in parse_reviews():
  user = review['user']
  if user not in users:
    users[user] = []
  users[user] += review['votes'].values()
for user_id, votes in users.items():
  print >> _OUTPUT_U, '%s,%f' % (user_id, std(votes))
  
