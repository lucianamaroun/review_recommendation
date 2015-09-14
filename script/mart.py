from pickle import dump, load
from numpy import nan, isnan
from sys import argv
from sklearn.ensemble import GradientBoostingRegressor

from src.modeling.modeling import model
from src.modeling.author_voter_modeling import model_author_voter_similarity, \
    model_author_voter_connection
from src.modeling.user_modeling import get_similar_users
from src.parsing.parser import parse_trusts
from src.util.evaluation import calculate_ndcg


_SAMPLE = float(argv[1])


#print 'Modeling'
#reviews, users, _, train, test = model(_SAMPLE)
#dump(reviews, open('pkl/reviews%f.pkl' % _SAMPLE, 'w'))
#dump(users, open('pkl/users%f.pkl' % _SAMPLE, 'w'))
#dump(train, open('pkl/train%f.pkl' % _SAMPLE, 'w'))
#dump(test, open('pkl/test%f.pkl' % _SAMPLE, 'w'))
#similar = get_similar_users(users)
#dump(similar, open('pkl/similar%f.pkl' % _SAMPLE, 'w'))
#trusts = parse_trusts()
#dump(trusts, open('pkl/trusts%f.pkl' % _SAMPLE, 'w'))
#sim_author_voter = model_author_voter_similarity(train, users, similar)
#dump(sim_author_voter, open('pkl/sim_author_voter%f.pkl' % _SAMPLE, 'w'))
#conn_author_voter = model_author_voter_connection(train, users, trusts)
#dump(conn_author_voter, open('pkl/conn_author_voter%f.pkl' % _SAMPLE, 'w'))
print 'Reading pickles'
reviews = load(open('pkl/reviews%f.pkl' % _SAMPLE, 'r'))
users = load(open('pkl/users%f.pkl' % _SAMPLE, 'r'))
train = load(open('pkl/train%f.pkl' % _SAMPLE, 'r'))
test = load(open('pkl/test%f.pkl' % _SAMPLE, 'r'))
similar = load(open('pkl/similar%f.pkl' % _SAMPLE, 'r'))
trusts = load(open('pkl/trusts%f.pkl' % _SAMPLE, 'r'))
sim_author_voter = load(open('pkl/sim_author_voter%f.pkl' % _SAMPLE, 'r'))
conn_author_voter = load(open('pkl/conn_author_voter%f.pkl' % _SAMPLE, 'r'))

print "Creating queries ids"
query = {}
count = 0
for vote in train:
  voter = vote['voter']
  if voter not in query:
    query[voter] = count
    count += 1 
for vote in test:
  voter = vote['voter']
  if voter not in query:
    query[voter] = count
    count += 1 

print "Creating average user (for mean imputation)"
avg_user = users.itervalues().next().copy()
for feature in avg_user:
  if feature in ['id', 'ratings', '_id']:
    continue
  avg_user[feature] = 0.0
  count = 0
  for user in users.itervalues():
    if not isnan(user[feature]):
      avg_user[feature] += user[feature]
      count += 1
  avg_user[feature] /= float(count)

review_feat = sorted(set(reviews[train[0]['review']].keys())
    .difference(set(['id', 'votes', '_id', 'product', 'category', 'text', 'user',
    'date'])))
user_feat = sorted(set(users[train[0]['reviewer']].keys())
    .difference(set(['id', '_id', 'ratings'])))

X_train = []
y_train = []
qid_train = []
for vote in train:
  X = []
  review = reviews[vote['review']]
  for feature in review_feat:
    X.append(review[feature])
  author = users[vote['reviewer']]
  for feature in user_feat:
    if isnan(author[feature]):
      X.append(avg_user[feature]) 
    else:
      X.append(author[feature])
  voter = users[vote['voter']]
  for feature in user_feat:
    if isnan(voter[feature]):
      X.append(avg_user[feature]) 
    else:
      X.append(voter[feature]) 
  X_train.append(X)
  y_train.append(vote['vote'])
  qid_train.append(query[vote['voter']])

X_test = []
y_test = []
qid_test = []
for vote in test:
  X = []
  review = reviews[vote['review']]
  for feature in review_feat:
    X.append(review[feature])
  author = users[vote['reviewer']] if vote['reviewer'] in users else avg_user
  for feature in user_feat:
    if isnan(author[feature]):
      X.append(avg_user[feature]) 
    else:
      X.append(author[feature])
  voter = users[vote['voter']] if vote['voter'] in users else avg_user
  for feature in user_feat:
    if isnan(voter[feature]):
      X.append(avg_user[feature]) 
    else:
      X.append(voter[feature]) 
  X_test.append(X)
  y_test.append(vote['vote'])
  qid_test.append(query[vote['voter']])

clf = GradientBoostingRegressor()# n_estimators=ntrees, learning_rate=0.1,
#max_depth=2, random_state=seed, loss='ls')

clf.fit(X_train , y_train)
pred = clf.predict(X_test)

output = open('out/mart%f.dat' % _SAMPLE, 'w')
for p in pred:
  print >> output, p
output.close()
