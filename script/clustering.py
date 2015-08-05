import pickle
import numpy

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

from src.parser import parse_trusts
from src.sampler import sample_reviews
from src.review_modeling import model_reviews_parallel
from src.user_modeling import model_users


# trusts = parse_trusts()
#trusts = pickle.load(open('pkl/trusts.pkl', 'r'))
#sel_reviews = sample_reviews(0.01)
#reviews = model_reviews_parallel(2, sel_reviews)
#reviews = pickle.load(open('pkl/reviews.pkl', 'r'))
#train, test = split_votes(reviews)
#train = pickle.load(open('pkl/train.pkl', 'r'))
#raw_users = model_users(reviews, train, trusts)
raw_users = pickle.load(open('pkl/users.pkl', 'r'))

users = {key: numpy.array([user['num_reviews'], user['avg_rating'],
    user['avg_help_rec'], user['avg_help_giv'], user['num_trustors'],
    user['num_trustees']])
    for (key, user) in raw_users.iteritems()}

X = numpy.array(users.values())
scaler = StandardScaler().fit(X)
X = scaler.transform(X)


for k in xrange(10, 100, 10):
  clustering = KMeans(k)
  clustering.fit(X)
  print 'KMeans, k = %d' % k
  print 'Sum of distances: %s' % clustering.inertia_
  silhouette = silhouette_score(X, clustering.labels_, metric='euclidean')
  print 'Silhouette: %s\n-----\n' % silhouette 

clustering = DBSCAN()
clustering.fit(X)
print 'DBScan, k = %d' % len(set([c for c in clustering.labels_ if c != -1]))
silhouette = silhouette_score(X, clustering.labels_, metric='euclidean')
print 'Silhouette: %s\n-----\n' % silhouette 

