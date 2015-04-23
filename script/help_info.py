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

user_votes = {}
user_reviews = {}
for review in parse_reviews():
  if not review['votes']:
    continue
  user = review['user']
  if user not in user_votes:
    user_votes[user] = []
    user_reviews[user] = 0
  user_votes[user] += review['votes'].values()
  user_reviews[user] += 1
print >> _OUTPUT_U, 'id,mean,sd,num_reviews'
for user_id, votes in user_votes.items():
  print >> _OUTPUT_U, '%s,%f,%f,%d' % (user_id, mean(votes), std(votes),
      user_reviews[user_id])
  
