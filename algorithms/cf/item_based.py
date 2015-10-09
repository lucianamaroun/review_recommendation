""" Item-Based Module
    -----------------

    Implementation of Item-Based Collaborative Filtering using cosine similarity
    and a parameterized neighbor of size K.

    Usage:
    $ python -m cf.item_based
"""


from math import sqrt
from pickle import load
from sys import argv, exit

from numpy import nan, isnan, zeros
from numpy.random import random

from algorithms.const import NUM_SETS, RANK_SIZE, REP
from evaluation.metrics import calculate_rmse, calculate_ndcg
from util.aux import sigmoid, sigmoid_der1, cosine, vectorize
from util.bias import BiasModel


K = 20
_PKL_DIR = 'out/pkl'
_VAL_DIR = 'out/val'
_OUTPUT_DIR = 'out/pred/'


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
    self.review_mean = None
    self.bias = None

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

#  def _calculate_review_mean(self, votes):
#    self.review_mean = {}
#    review_count = {}
#    for vote in votes:
#      review = vote['review']
#      if review not in self.review_mean:
#        self.review_mean[review] = 0.0
#        review_count[review] = 0.0
#      self.review_mean[review] += vote['vote']
#      review_count[review] += 1.0
#    for review in self.review_mean:
#      self.review_mean[review] /= review_count[review]
  
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
        r != review]
    if len(selected_reviews) < self.K:
      return selected_reviews
    neighbors = []
    for _ in xrange(self.K): # selecting K top, instead of sorting
      best_i = 0
      best = selected_reviews[best_i]
      for i in xrange(1, len(selected_reviews)):
        r = selected_reviews[i]
        if r and (not best or self.Sim[review][r] > self.Sim[review][best]):
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
   # self._calculate_review_mean(votes)
    self._compute_similarity()
    self.bias = BiasModel()
    new_votes = self.bias.fit_transform(votes)
    self._initialize_matrix(new_votes)
  
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
      pred += self.Sim[review][n] * \
          (self.Reviews[n][user] - self.review_mean[n])
      sim_total += self.Sim[review][n]
    pred = self.review_mean[review] + pred / sim_total if sim_total > 0 else nan
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
    return pred


if __name__ == '__main__':
  for i in xrange(1):#NUM_SETS):
    print 'Reading pickles'
    train = load(open('%s/train-%d.pkl' % (_PKL_DIR, i), 'r'))
    val = load(open('%s/validation-%d.pkl' % (_PKL_DIR, i), 'r'))
    test = load(open('%s/test-%d.pkl' % (_PKL_DIR, i), 'r'))
    reviews = load(open('%s/reviews-%d.pkl' % (_PKL_DIR, i), 'r'))
    truth = [v['vote'] for v in train]
    overall_avg = float(sum(truth)) / len(train)
    
    for j in xrange(REP):
      print 'Fitting Model'
      model = ItemBasedModel()
      model.fit(train)

      print 'Calculate Predictions'
      pred = model.predict(train)
       
      print 'TRAINING ERROR'
      print '-- RMSE: %f' % calculate_rmse(pred, truth)
      pred_group = {}
      truth_group = {}
      for vote in train:
        voter = vote['voter']
        product = reviews[vote['review']]['product']
        key = (voter, product)
        if key not in pred_group:
          pred_group[key] = []
          truth_group[key] =[]
        pred_group[key].append(pred[i])
        truth_group[key].append(truth[i])
      score_sum = 0.0
      for key in pred_group:
        ndcg = calculate_ndcg(pred_group[key], truth_group[key], RANK_SIZE)
        score_sum += ndcg
      score = score_sum / len(pred_group)
      print '-- nDCG@%d: %f' % (RANK_SIZE, score)
      
      pred = model.predict(val) 
      print 'Outputting validation prediction'
      output = open('%s/ib-%d-%d.dat' % (_VAL_DIR, i, j), 'w')
      for p in pred:
        print >> output, overall_avg if isnan(p) else p
      output.close()
      
      pred = model.predict(test) 
      print 'Outputting testing prediction'
      output = open('%s/ib-%d-%d.dat' % (_OUTPUT_DIR, i, j), 'w')
      for p in pred:
        print >> output, overall_avg if isnan(p) else p
      output.close()
