""" Item-Based Module
    -----------------

    Implementation of Item-Based Collaborative Filtering using cosine similarity
    and a parameterized neighbor of size K.

    Usage:
    $ python -m cf.item_based
"""


from math import sqrt

from numpy import nan, isnan, zeros
from numpy.random import random

from cap.aux import sigmoid, sigmoid_der1
from src.modeling.modeling import _SAMPLE_RATIO
from src.util.aux import cosine, vectorize


K = 20
_SAMPLE = 0.001


class ItemBasedModel(object):
  """ Class implementing an Item-Based Collaborative Filtering Model. """

  def __init__(self, k=K):
    """ Initializes attributes. 

        Args:
          k (optional): size of the neighborhood to consider when finding
        similar items.

        Returns:
          None.
    """
    self.Reviews = {} # Dictionary of reviews' votes 
    self.Users = set() # Set of users in train (voted previously by anyone)
    self.Sim = {} # Dictionary of reviews's similarities
    self.K = k # Neighborhood size


  def _initialize_matrix(self, votes):
    """ Initializes sparse matrix of votes, represented as a dictionary, and
        auxiliar structures.

        Args:
          votes: list of votes (training set).
        
        Returns:
          None. Instance fields are updated.
    """
    for vote in votes:
      user = vote['voter']
      review = vote['review']
      if review not in self.Reviews:
        self.Reviews[review] = {}
      self.Reviews[review][user] = vote['vote']
      self.Users.add(user)
  
  def _compute_similarity(self):
    """ Computes the similarity between every pair of items. Cosine similarity
        is used.

        Args:
          None.

        Returns:
          None. Similarity dictionary field is updated.
    """
    for review in self.Reviews:
      self.Sim[review] = {}
    for review in self.Reviews:
      for other in [r for r in self.Reviews if r > review]:
        r_vec, o_vec = vectorize(self.Reviews[review], self.Reviews[other])
        similarity = cosine(r_vec, o_vec)
        self.Sim[review][other] = similarity 
        self.Sim[other][review] = similarity 
    
  def _compute_neighbors(self, user, review):
    """ Compute the K nearest neighbors of a given review which have been
        evaluated by a given user.

        Args:
          user: the user id.
          review: the review id.

        Returns:
          A list of K review ids, the neighbors.
    """
    selected_reviews = [r for r in self.Reviews if user in self.Reviews[r] and 
        r != reviews]
    neighbors = []
    for _ in xrange(self.K): # selecting K top, instead of sorting
      best_i = 0
      best = selected_review[best_i]
      for i in xrange(1, len(selected_reviews)):
        r = selected_reviews[i]
        if r and self.Sim[review][r] > self.Sim[review][best]:
          best = r
          best_i = i
      neighbors.append(best)
      selected_reviews[best_i] = None
    return neighbors

  def fit(self, votes):
    """ Fits an Item-based Collaborative Filtering model given training set 
        (votes). Basically, similarities are calculated and users and reviews,
        accounted for.

        Args:
          vote: list of votes, represented as dictionaries (training set).
        
        Returns:
          None. Instance fields are updated.
    """
    self._initialize_matrix(votes)
    self._compute_similarity()
  
  def _calculate_prediction(self, user, review):
    """ Calculates a single prediction for a given (user, review) pair.
    
        Args:
          user: the user id.
          review: the review id.

        Returns:
          A real value with the predicted vote.
    """
    neighbors = self._compute_neighbors(user, review)
    pred = 0
    sim_total = 0
    for n in neighbors:
      pred += self.Sim[user][n] * self.Users[n][review]
      sim_total += self.Sim[user][n]
    pred /= sim_total
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
        pred.append(self._calculate_prediction(u, r, neighbors))
      else:
        pred.append(nan)
    return pred


if __name__ == '__main__':
  import pickle
  print 'Reading pickles'
 # _, _, _, train, test = model()
 # pickle.dump(train, open('pkl/cap_train%f.pkl' % _SAMPLE, 'w'))
 # pickle.dump(test, open('pkl/cap_test%f.pkl' % _SAMPLE, 'w'))
  train = pickle.load(open('pkl/cap_train%f.pkl' % _SAMPLE, 'r'))
  test = pickle.load(open('pkl/cap_test%f.pkl' % _SAMPLE, 'r'))
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
  
  pred = model.predict(test) 
  print 'TESTING ERROR'
  sse = sum([(overall_avg if isnan(pred[i]) else 
      pred[i] - test[i]['vote']) ** 2 for i in xrange(len(test))])
  rmse = sqrt(sse/len(test))
  print 'RMSE: %s' % rmse

