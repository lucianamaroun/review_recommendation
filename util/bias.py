""" Bias Baseline 
    -------------

    Implementation of bias baseline, in which the prediction is composed by a
    sum of overall average, user bias and item bias, the last two being
    variables to be fitted.

    Usage:
    $ python -m util.bias
"""


from math import sqrt
from sys import argv, exit

from numpy import nan, isnan
from numpy.random import random
from pickle import load

from evaluation.metrics import calculate_rmse, calculate_ndcg


_ITER = 1000      # number of iterations of stochastic gradient descent
_ALPHA = 0.01     # starting learning rate (MOGHADDAM, with update)
_BETA = 0.01      # regularization factor (MOGHADDAM)
_SAMPLE = 0.001 
_TOL = 1e-6
_PKL_DIR = 'out/pkl'
_OUTPUT_DIR = 'out/pred'

class BiasModel(object):
  """ Class implementing a Bias Baseline Model. """

  def __init__(self):
    """ Discriminates existing attributes, initilizing all to None.

        Args:
          None.

        Returns:
          None.
    """
    self.item_bias = None
    self.user_bias = None
    self.overall_mean = None

  def _initialize(self, votes):
    """ Initializes bias values for each entity and calculates overall average.

        Observation:
        - Instead of using random values, static estimation is used by averaging
          deviation from mean (or mean + previous bias).

        Args:
          votes: list of votes (training set).
        
        Returns:
          None. Instance fields are updated.
    """
    self.user_bias = {}
    user_count = {}
    self.item_bias = {}
    item_count = {}
    self.overall_mean = 0
    count = 0
    for vote in votes:
      self.overall_mean += vote['vote']
      count += 1
    self.overall_mean /= float(count)
    for vote in votes:
      user = vote['voter']
      if user not in self.user_bias:
        self.user_bias[user] = 0
        user_count[user] = 0
      self.user_bias[user] += (vote['vote'] - self.overall_mean)
      user_count[user] += 1
    for user in self.user_bias:
      self.user_bias[user] /= float(user_count[user])
    for vote in votes:
      user = vote['voter']
      item = vote['review']
      if item not in self.item_bias:
        self.item_bias[item] = 0
        item_count[item] = 0
      self.item_bias[item] += (vote['vote'] - self.overall_mean -
          self.user_bias[user]) 
      item_count[item] += 1
    for item in self.item_bias:
      self.item_bias[item] /= float(item_count[item])
    
  def fit(self, votes):
    """ Fits a Bias Baseline model given training set (votes).

        Args:
          vote: list of votes, represented as dictionaries (training set).
        
        Returns:
          None. Instance fields are updated.
    """
    self._initialize(votes)
    previous = float('inf')
    for _ in xrange(_ITER):
      for vote in votes:
        u = vote['voter']
        i = vote['review']
        pred = self.overall_mean + self.user_bias[u] + self.item_bias[i]
        error = float(vote['vote']) - pred 
        self.user_bias[u] += _ALPHA * 2 * (error - _BETA * self.user_bias[u])
        self.item_bias[i] += _ALPHA * 2 * (error - _BETA * self.item_bias[i])
      value = 0.0
      for vote in votes:
        u = vote['voter']
        i = vote['review']
        pred = self.overall_mean + self.user_bias[u] + self.item_bias[i]
        value += (vote['vote'] - pred) ** 2
      for u in self.user_bias:
        value += self.user_bias[u] ** 2
      for i in self.item_bias:
        value += self.item_bias[i] ** 2
      if abs(previous - value) < _TOL:
        print 'Convergence'
        break
      previous = value

  def fit_transform(self, votes):
    """ Fits Bias Model on data and adjust vote values to contain only unbiased
        value.

        Args:
          votes: list of votes to learn model and transform values.

        Returns:
          None. Changes are made in place.
    """
    self.fit(votes)
    return self.transform(votes)

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
      u = vote['voter']
      i = vote['review']
      if u in self.user_bias and i in self.item_bias:
        dot = self.overall_mean + self.user_bias[u] + self.item_bias[i] 
        pred.append(dot)
      else:
        pred.append(nan)
    return pred

  def get_user_bias(self, user):
    """ Gets a bias regarding a user on a fitted bias model.

        Args:
          user: the id of the user.

        Returns:
          A float with user bias, in warm-start case, or nan, in cold-start.
    """
    return self.user_bias[user] if user in self.user_bias else nan
  
  def get_item_bias(self, item):
    """ Gets a bias regarding an item on a fitted bias model.

        Args:
          item: the id of the item.

        Returns:
          A float with item bias, in warm-start case, or nan, in cold-start.
    """
    return self.item_bias[item] if item in self.item_bias else nan
 
  def transform(self, votes):
    """ Removes bias values and overall mean from votes.

        Args:
          votes: list of votes to remove bias; it should be the same list
        fitted, because all entities have to have a bias associated.

        Returns:
          A new list of votes with unbiased values. 
    """
    new_votes = []
    for vote in votes:
      new_vote = vote.copy()
      u = vote['voter']
      i = vote['review']
      new_vote['vote'] -= self.overall_mean + self.user_bias[u] + \
          self.item_bias[i]
      new_votes.append(new_vote) 
    return new_votes

  def add_bias(self, votes, pred):
    """ Removes bias values and overall mean from votes.

        Args:
          votes: votes whose values are being predicted.
          pred: unbiased estimates of votes.

        Returns:
          None. The pred list is changed in place.
    """
    for index, vote in enumerate(votes):
      u = vote['voter']
      i = vote['review']
      if isnan(pred[index]):
        continue
      pred[index] += self.overall_mean
      if u in self.user_bias:
        pred[index] += self.user_bias[u]
      if i in self.item_bias:
        pred[index] += self.item_bias[i]


if __name__ == '__main__':
  print 'Reading pickles'
  train = load(open('%s/train%.2f.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
  test = load(open('%s/test%.2f.pkl' % (_PKL_DIR, _SAMPLE * 100), 'r'))
  overall_mean = float(sum([float(v['vote']) for v in train])) / len(train)
  
  print 'Fitting Model'
  model = BiasModel()
  model.fit(train)

  print 'Calculating Predictions'
  pred = model.predict(train)
   
  print 'TRAINING ERROR'
  truth = [v['vote'] for v in train]
  rmse = calculate_rmse(pred, truth) 
  print 'RMSE: %s' % rmse
  for i in xrange(5, 21, 5):
    score = calculate_ndcg(pred, truth, i)
    print 'NDCG@%d: %f' % (i, score)
  
  print 'Outputing Prediction'
  pred = model.predict(test) 
  output = open('%s/bias%.2f.dat' % (_OUTPUT_DIR, _SAMPLE * 100), 'w')
  for p in pred:
    print >> output, overall_mean if isnan(p) else p
  output.close()

