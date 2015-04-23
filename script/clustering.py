from src import parser
from src import sampler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from src.user_modeling import model_users

trusts = parser.parse_trusts()
sel_reviews = sampler.sample_reviews(0.01)
reviews = model_reviews_parallel(2, sel_reviews)
train, test = split_votes(reviews)
raw_users = model_users(reviews, train, trusts)

users = {key: [user['num_reviews'], user['avg_rating'], user['avg_help_rec'],
    user['avg_help_giv'], user['num_trustors'], user['num_trustees']]
    for (key, user) in raw_users.iteritems()}

clustering = KMeans(5)
clustering.fit(users.values())
print 'Sum of distances: %s\n' % clustering.inertia_
silhouette = silhouette_score(users.values(), clustering.labels_,
    metric='euclidian')
print ': %s\n' % silhouette 

