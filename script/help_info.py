""" Review sd script
    ----------------

    Generates and output reviews' standard deviation of votes.

    Usage:
      $ python -m script.help_info
"""


from numpy import mean, std

from src.parser import parse_reviews


_OUTPUT_R = open('out/review_help_info.csv', 'w')
_OUTPUT_U = open('out/user_help_info.csv', 'w')


print >> _OUTPUT_R, 'id,mean,sd'
for review in parse_reviews():
  votes = review['votes'].values()
  if not votes:
    continue
  print >> _OUTPUT_R, '%d,%f,%f' % (review['id'], mean(votes), std(votes))

users = {}
for review in parse_reviews():
  if not review['votes']:
    continue
  user = review['user']
  if user not in users:
    users[user] = []
  users[user] += review['votes'].values()
print >> _OUTPUT_U, 'id,mean,sd'
for user_id, votes in users.items():
  print >> _OUTPUT_U, '%s,%f,%f' % (user_id, mean(votes), std(votes))
  
