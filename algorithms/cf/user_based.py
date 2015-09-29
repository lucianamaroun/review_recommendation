""" User-based Module 
    -----------------

    Implementation of User-Based Collaborative Filtering, using cosine
    similarity and fixed neighborhood of parameterized size K.

    Usage:
    $ python -m cf.user_based
"""


from math import sqrt
from sys import argv

from numpy import nan, isnan, zeros
from numpy.random import random

from util.aux import sigmoid, sigmoid_der1, cosine, vectorize
from util.bias import BiasModel


if len(argv) != 2:
  import sys
  sys.exit()


K = 20
_SAMPLE = float(argv[1])
_PKL_DIR = 'out/pkl'
_VAL_DIR = 'out/val'
_OUTPUT_DIR = 'out/pred'


class UserBasedModel(object):
  """ Class implementing a User-Based Collaborative Filtering Module. """ 

  def __init__(self, k=K):
    """ Initializes empty attributes and K. 

        Args:
          k (optional): the size of the neighborhood. If none is given, a
        parameterized default is used.

        Returns:
          None.
    """
    self.Users = {}       # Dictionary of users' votes 
    self.Reviews = set()  # Set of reviews in train (voted previously by anyone)
    self.Sim = {}         # Dictionary of users's similarities
    self.K = k            # Neighborhood size
    self.voter_mean = None
    self.bias = None

  def _initialize_matrix(self, votes):
    """ Initializes matrices and mappings given votes. Each entity id is mapped
        to an index in a dimension of  the matrix.

        Args:
          votes: list of votes (training set).
        
        Returns:
          None. Instance fields are updated.
    """
    for vote in votes:
      user = vote['voter']
      review = vote['review']
      if user not in self.Users:
        self.Users[user] = {}
      self.Users[user][review] = vote['vote']
      self.Reviews.add(review)
  
#  def _calculate_voter_mean(self, votes):
#    self.voter_mean = {}
#    voter_count = {}
#    for vote in votes:
#      voter = vote['voter']
#      if voter not in self.voter_mean:
#        self.voter_mean[voter] = 0.0
#        voter_count[voter] = 0.0
#      self.voter_mean[voter] += vote['vote']
#      voter_count[voter] += 1.0
#    for voter in self.voter_mean:
#      self.voter_mean[voter] /= voter_count[voter]

  def _compute_similarity(self):
    """ Computes the similarity between every pair of users, using cosine.

        Args:
          None.

        Returns:
          None. The similarity dictionary field is updated.
    """
    for user in self.Users:
      self.Sim[user] = {}
    for user in self.Users:
      for other in [u for u in self.Users if u > user]:
        u_vec, o_vec = vectorize(self.Users[user], self.Users[other])
        similarity = cosine(u_vec, o_vec)
        self.Sim[user][other] = similarity 
        self.Sim[other][user] = similarity 
    
  def _compute_neighbors(self, user, review):
    """ Computes the K nearest neighbors of a user which have voted in a review.

        Observation:
        - Instead of performing sorting of the list of users by similarity, the
          closest neighbor is selected K times, yielding complexity O(knm)
          instead of O(n log n m). TODO: REALLY BETTER?

        Args:
          user: the user id.
          review: the review id.

        Returns:
          A list with K user ids, the neighbors.
    """
    selected_users = [u for u in self.Users if review in self.Users[u] and u !=
        user]
    if len(selected_users) < self.K:
      return selected_users
   # neighbors = sorted(selected_users, key=lambda u: self.Sim[user][u], 
   #     reverse=True)[:self.K]
    neighbors = []
    for _ in xrange(self.K): # selecting K top, instead of sorting
      best_i = 0
      best = selected_users[best_i]
      for i in xrange(1, len(selected_users)):
        u = selected_users[i]
        if u and (not best or self.Sim[user][u] > self.Sim[user][best]):
          best = u
          best_i = i
      neighbors.append(best)
      selected_users[best_i] = None
    return neighbors

  def fit(self, votes):
    """ Fits a User-Based model given training set (votes).

        Args:
          vote: list of votes, represented as dictionaries (training set).
        
        Returns:
          None. Instance fields are updated.
    """
    self._initialize_matrix(votes)
    #self._calculate_voter_mean(votes)
    self._compute_similarity()
    self.bias = BiasModel()
    new_votes = self.bias.fit_transform(votes)
    print new_votes[:10]
    self._initialize_matrix(new_votes)

  def _calculate_prediction(self, user, review):
    """ Calculates a single prediction for a given (user, review) pair.
    
        Args:
          user: the user id.
          review: the review id.

        Returns:
          A real value with the predicted vote.
    """
    pred = 0
    sim_total = 0
    neighbors = self._compute_neighbors(user, review)
    for n in neighbors:
      pred += self.Sim[user][n] * \
          self.Users[n][review] # - self.voter_mean[n])
      sim_total += self.Sim[user][n]
   # pred = self.voter_mean[user] + pred / sim_total if sim_total > 0 else nan
    pred = pred / sim_total if sim_total > 0 else nan
    return pred

  def predict(self, votes):
    """ Predicts a set of vote examples using previous fitted model.

        Args:
          votes: list of dictionaries, representing votes, to predict
        helpfulness vote value.

        Returns:
          A list of floats with predicted vote values.
    """
    pred = []
    for vote in votes:
      u = vote['voter'] if vote['voter'] in self.Users else -1
      r = vote['review'] if vote['review'] in self.Reviews else -1
      if u != -1 and r != -1:
        pred.append(self._calculate_prediction(u, r))
      else:
        pred.append(nan)
    self.bias.add_bias(votes, pred)
    return pred


if __name__ == '__main__':
  import pickle
  print 'Reading pickles'
 # _, _, _, train, test = model()
 # pickle.dump(train, open('pkl/train%f.pkl' % _SAMPLE, 'w'))
 # pickle.dump(test, open('pkl/test%f.pkl' % _SAMPLE, 'w'))
  train = pickle.load(open('%s/train%.2f.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
  val = pickle.load(open('%s/validation%.2f.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
  test = pickle.load(open('%s/test%.2f.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
  overall_avg = float(sum([float(v['vote']) for v in train])) \
      / len(train)


  print 'Fitting Model'
  model = UserBasedModel()
  model.fit(train)

  print 'Calculate Predictions'
  pred = model.predict(train)

  print 'TRAINING ERROR'
  sse = sum([(overall_avg if isnan(pred[i]) else 
      pred[i] - train[i]['vote']) ** 2 for i in xrange(len(train))])
  rmse = sqrt(sse/len(train))
  print 'RMSE: %s' % rmse
  
  pred = model.predict(val) 
  print 'Outputting validation prediction'
  output = open('%s/ub%.2f.dat' % (_VAL_DIR, _SAMPLE * 100), 'w')
  for p in pred:
    print >> output, overall_avg if isnan(p) else p
  output.close()
  
  pred = model.predict(test) 
  print 'Outputting testing prediction'
  output = open('%s/ub%.2f.dat' % (_OUTPUT_DIR, _SAMPLE * 100), 'w')
  for p in pred:
    print >> output, overall_avg if isnan(p) else p
  output.close()
