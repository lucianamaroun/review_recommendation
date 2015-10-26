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

from perf.metrics import calculate_rmse, calculate_ndcg


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
    self.product_bias = None
    self.author_bias = None
    self.voter_bias = None
    self.overall_mean = None

  def _initialize(self, votes, reviews):
    """ Initializes bias values for each entity and calculates overall average.

        Observation:
        - Instead of using random values, static estimation is used by averaging
          deviation from mean (or mean + previous bias).

        Args:
          votes: list of votes (training set).
          reviews: dictionary of reviews, to obtain product.
        
        Returns:
          None. Instance fields are updated.
    """
    self.voter_bias = {}
    voter_count = {}
    self.author_bias = {}
    author_count = {}
    self.product_bias = {}
    product_count = {}
    self.overall_mean = 0
    count = 0
    for vote in votes:
      self.overall_mean += vote['vote']
      count += 1
    self.overall_mean /= float(count)
    for vote in votes:
      voter = vote['voter']
      if voter not in self.voter_bias:
        self.voter_bias[voter] = 0
        voter_count[voter] = 0
      self.voter_bias[voter] += (vote['vote'] - self.overall_mean)
      voter_count[voter] += 1
    for voter in self.voter_bias:
      self.voter_bias[voter] /= float(voter_count[voter])
    for vote in votes:
      voter = vote['voter']
      author = vote['author']
      if author not in self.author_bias:
        self.author_bias[author] = 0
        author_count[author] = 0
      self.author_bias[author] += (vote['vote'] - self.overall_mean -
          self.voter_bias[voter]) 
      author_count[author] += 1
    for author in self.author_bias:
      self.author_bias[author] /= float(author_count[author])
    for vote in votes:
      voter = vote['voter']
      author = vote['author']
      product = reviews[vote['review']]['product']
      if product not in self.product_bias:
        self.product_bias[product] = 0
        product_count[product] = 0
      self.product_bias[product] += (vote['vote'] - self.overall_mean -
          self.voter_bias[voter] - self.author_bias[author]) 
      product_count[product] += 1
    for product in self.product_bias:
      self.product_bias[product] /= float(product_count[product])
    
  def fit(self, votes, reviews):
    """ Fits a Bias Baseline model given training set (votes).

        Args:
          vote: list of votes, represented as dictionaries (training set).
          reviews: dictionary of reviews.
        
        Returns:
          None. Instance fields are updated.
    """
    self._initialize(votes, reviews)
    previous = float('inf')
    for _ in xrange(_ITER):
      for vote in votes:
        voter = vote['voter']
        author = vote['author']
        product = reviews[vote['review']]['product']
        pred = self.overall_mean + self.voter_bias[voter] + \
            self.author_bias[author] + self.product_bias[product]
        error = pred - float(vote['vote']) 
        # one does not depend on the other: no need for temporary variables
        self.voter_bias[voter] -= _ALPHA * 2 * (error + _BETA *
            self.voter_bias[voter])
        self.author_bias[author] -= _ALPHA * 2 * (error + _BETA * \
            self.author_bias[author])
        self.product_bias[product] -= _ALPHA * 2 * (error + _BETA * \
            self.product_bias[product])
      value = 0.0
      for vote in votes:
        voter = vote['voter']
        author = vote['author']
        product = reviews[vote['review']]['product']
        pred = self.overall_mean + self.voter_bias[voter] + \
            self.author_bias[author] + self.product_bias[product]
        value += (vote['vote'] - pred) ** 2
      for voter in self.voter_bias:
        value += self.voter_bias[voter] ** 2
      for author in self.author_bias:
        value += self.author_bias[author] ** 2
      for product in self.product_bias:
        value += self.product_bias[product] ** 2
      if abs(previous - value) < _TOL:
        print 'Convergence'
        break
      previous = value

  def fit_transform(self, votes, reviews):
    """ Fits Bias Model on data and adjust vote values to contain only unbiased
        value.

        Args:
          votes: list of votes to learn model and transform values.
          reviews: dictionary of reviews.

        Returns:
          None. Changes are made in place.
    """
    self.fit(votes, reviews)
    return self.transform(votes, reviews)

  def predict(self, votes, reviews):
    """ Predicts a set of vote examples using previous fitted model.

        Args:
          votes: list of dictionaries, representing votes, to predict
        helpfulness vote value.
          reviews: dictionary of reviews.

        Returns:
          A list of floats with predicted vote values.
    """
    pred = []
    for vote in votes:
      voter = vote['voter']
      author = vote['author']
      product = reviews[vote['review']]['product'] 
      value = self.overall_mean
      if voter in self.voter_bias: # if no bias, we assume 0
        dot += self.voter_bias[voter]
      if author in self.author_bias:
        dot += self.author_bias[author]
      if product in self.product_bias:
        dot += self.product_bias[product]
      pred.append(value)
    return pred

  def transform(self, votes, reviews):
    """ Removes bias values and overall mean from votes.

        Args:
          votes: list of votes to remove bias; it should be the same list
        fitted, because all entities have to have a bias associated.
          reviews: dictionary of reviews.

        Returns:
          A new list of votes with unbiased values. 
    """
    new_votes = []
    for vote in votes:
      new_vote = vote.copy()
      voter = vote['voter']
      author = vote['author']
      product = reviews[vote['review']]['product']
      new_vote['vote'] -= self.overall_mean + self.voter_bias[voter] + \
          self.author_bias[author] + self.product_bias[product]
      new_votes.append(new_vote) 
    return new_votes

  def add_bias(self, votes, reviews, pred):
    """ Removes bias values and overall mean from votes.

        Args:
          votes: votes whose values are being predicted.
          reviews: dictionary of reviews.
          pred: unbiased estimates of votes.

        Returns:
          None. The pred list is changed in place.
    """
    for index, vote in enumerate(votes):
      voter = vote['voter']
      author = vote['author']
      product = reviews[vote['review']]['product']
      if isnan(pred[index]):
        continue
      pred[index] += self.overall_mean
      if voter in self.voter_bias:
        pred[index] += self.voter_bias[voter]
      if author in self.author_bias:
        pred[index] += self.author_bias[author]
      if product in self.product_bias:
        pred[index] += self.product_bias[product]


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

