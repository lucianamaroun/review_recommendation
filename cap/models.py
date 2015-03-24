import numpy as np
from scipy.stats import logistic

from cap import constants


class Parameter(object):
  """ Class specifying a Parameter, which defines latent variables distributions
      and relationship with observed variables.  
  """

  def __init__(self, name, size):
    """ Constructor of Parameter. The values are randomly initialized using
        Uniform(0, 1).

        Args:
          name: string with the name of the parameter.
          size: pair with vector or matrix dimensions.

        Returns:
          None.
    """
    self.name = name
    self.size = size
    if type(size) is tuple:
      self.value = np.array([[np.random.random() + 0.0000001 for _ in
          range(size[1])] for _ in range(size[0])])
    else:
      if size == 1:
        self.value = np.random.random() + 0.000001
      else:
        self.value = np.array([np.random.random() + 0.000001 for _ in
            range(size)])
    np.reshape(self.value, size)


class ParameterCollection(object):
  """ Class specifying the collection of parameters of the model. Each parameter
      is singularly defined, that is, there is only one for all the instances.
  """

  def __init__(self):
    """ Constructor of ParameterCollection.
    
        Args:
          None.
        
        Returns:
          None.
    """
    self.g = Parameter('g', (1, 17))
    self.d = Parameter('d', (1, 9))
    self.b = Parameter('b', (1, 5))
    self.r = Parameter('r', (1, 7))
    self.h = Parameter('h', (1, 4))#5))
    self.W = Parameter('W', (constants.K, 9))
    self.V = Parameter('V', (constants.K, 17))
    self.var_beta = Parameter('var_beta', 1)
    self.var_alpha = Parameter('var_alpha', 1)
    self.var_xi = Parameter('var_xi', 1)
    self.var_gamma = Parameter('var_gamma', 1)
    self.var_lambda = Parameter('var_lambda', 1)
    self.var_u = Parameter('var_u', (constants.K, constants.K))
    self.var_v = Parameter('var_v', (constants.K, constants.K))
    self.var_H = Parameter('var_H', 1)


class Variable(object):
  """ Class defining a latent variable.
  """
  
  def __init__(self, name, size, coef, var, features): 
    """ Constructor of Variable.
    
        Args:
          name: string with the name of the variable.
          size: pair with the dimensions of the variable.
          coef: name of the coefficients multiplying features.
          var: name of the variance parameter.
          features: set of observed features associated with the variable.
          
        Returns:
          None.
    """
    self.name = name
    self.size = size 
    self.coef = coef 
    self.var = var 
    self.features = np.reshape(features, (features.shape[0], 1))
    self.value = np.reshape(np.zeros(size), (size, 1)) if size > 1 else 0 
    self.samples = []
    self.e_mean = 0
    self.e_var = 0

  def update_value(self, value):
    self.value = np.reshape(value, (self.size, 1)) if self.size > 1 else 0 


class VariableCollection(object):
  """ Class defining a collection of variables composing a model.
  """

  def __init__(self):
    """ Constructor of VariableCollection.

        Args:
          None. The variables are added through methods.

        Returns:
          None.
    """
    self.beta = {}
    self.alpha = {}
    self.xi = {}
    self.gamma = {}
    self.lambd = {}
    self.u = {}
    self.v = {}

  def add_beta_variable(self, review_id, features):
    """ Adds a beta variable associated with a review.
    
        Args:
          review_id: string with the id of the review.
          features: a np.array with size (17, 1) with review content features.
        
        Returns:
          None.
    """
    self.beta[review_id] = Variable('beta', 1, 'g', 'var_beta', features)

  def add_alpha_variable(self, voter_id, features):
    """ Adds an alpha variable associated with a voter.
    
        Args:
          voter_id: string with the id of the voter.
          features: a np.array with size (9, 1) with voter individual features.
        
        Returns:
          None.
    """
    self.alpha[voter_id] = Variable('alpha', 1, 'd', 'var_alpha', features)

  def add_xi_variable(self, author_id, features):
    """ Adds a xi variable associated with an author.
    
        Args:
          author_id: string with the id of the author.
          features: a np.array with size (5, 1) with author individual features.
        
        Returns:
          None.
    """
    self.xi[author_id] = Variable('xi', 1, 'b', 'var_xi', features)

  def add_gamma_variable(self, author_voter, features):
    """ Adds a gamma variable associated with a pair of author and voter.
    
        Args:
          author_voter: pair of strings with the id of the author and the id of
            the voter, respectively.
          features: a np.array with size (7, 1) with author and voter similarity
            features.
        
        Returns:
          None.
    """
    self.gamma[(author_voter)] = Variable('gamma', 1, 'r', 'var_gamma',
        features)

  def add_lambda_variable(self, author_voter, features):
    """ Adds a lambda variable associated with a pair of author and voter.
    
        Args:
          author_voter: pair of strings with the id of the author and the id of
            the voter, respectively.
          features: a np.array with size (5, 1) with author and voter connection
            features.
        
        Returns:
          None.
    """
    self.lambd[(author_voter)] = Variable('lambda', 1, 'h', 'var_lambda',
        features)

  def add_u_variable(self, voter_id, features):
    """ Adds a u lantent vector associated with a voter.
    
        Args:
          voter_id: string with the id of the voter.
          features: a np.array with size (9, 1) with voter individual features.
        
        Returns:
          None.
    """
    self.u[voter_id] = Variable('u', constants.K, 'W', 'var_u', features)


  def add_v_variable(self, review_id, features):
    """ Adds a v lantent vector associated with a review.
    
        Args:
          review_id: string with the id of the review.
          features: a np.array with size (17, 1) with review content features.
        
        Returns:
          None.
    """
    self.v[review_id] = Variable('v', constants.K, 'V', 'var_u', features)

  def get_alpha_mean_and_variance(self, voter_id, votes, parameters):
    """ Calculates mean and variance of probability distribution of [alpha|Rest].
    
        Args:
          voter_id: the id of voter to index alpha.
          votes: list of vote dictionaries whose voter was voter_id.
          parameters: a ParameterCollection object with the current parameters of
            the model.

        Returns:
          A pair with the calculated mean and variance.
    """
    variance = 0
    mean = 0
    for vote in votes:
      truth = vote['vote']
      review_id = vote['review']
      author_id = vote['reviewer']
      rest = truth - self.beta[review_id].value - self.xi[author_id].value - \
          self.u[voter_id].value.T.dot(self.v[review_id].value)
      rest -= self.gamma[(author_id, voter_id)].value if (author_id, voter_id) \
          in self.gamma else 0
      rest -= self.lambd[(author_id, voter_id)].value if (author_id, voter_id) \
          in self.lambd else 0
      variance += 1/parameters.var_H.value
      mean += rest/parameters.var_H.value
    variance = 1 / (1/parameters.var_alpha.value + variance)
    mean += parameters.d.value.dot(self.alpha[voter_id].features) / \
        parameters.var_alpha.value
    mean *= variance
    return mean, variance

  def get_beta_mean_and_variance(self, review_id, votes, parameters):
    """ Calculates mean and variance of probability distribution of [beta|Rest].
    
        Args:
          review_id: the id of voter to index beta.
          votes: list of vote dictionaries associated with the review.
          parameters: a ParameterCollection object with the current parameters of
            the model.

        Returns:
          A pair with the calculated mean and variance.
    """
    variance = 0
    mean = 0
    for vote in votes:
      truth = vote['vote']
      voter_id = vote['voter']
      author_id = vote['reviewer']
      rest = truth - self.alpha[voter_id].value - self.xi[author_id].value - \
          self.u[voter_id].value.T.dot(self.v[review_id].value)
      rest -= self.gamma[(author_id, voter_id)].value if (author_id, voter_id) \
          in self.gamma else 0
      rest -= self.lambd[(author_id, voter_id)].value if (author_id, voter_id) \
          in self.lambd else 0
      variance += 1/parameters.var_H.value
      mean += rest/parameters.var_H.value
    variance = 1 / (1/parameters.var_beta.value + variance)
    mean += parameters.g.value.dot(self.beta[review_id].features) / \
        parameters.var_beta.value
    mean *= variance
    return mean, variance
    
  def get_xi_mean_and_variance(self, author_id, votes, parameters):
    """ Calculates mean and variance of probability distribution of [xi|Rest].
    
        Args:
          author_id: the id of voter to index xi.
          votes: list of vote dictionaries whose author is author_id.
          parameters: a ParameterCollection object with the current parameters of
            the model.

        Returns:
          A pair with the calculated mean and variance.
    """
    variance = 0
    mean = 0
    for vote in votes:
      truth = vote['vote']
      review_id = vote['review']
      voter_id = vote['voter']
      rest = truth - self.alpha[voter_id].value - self.beta[review_id].value - \
          self.u[voter_id].value.T.dot(self.v[review_id].value)
      rest -= self.gamma[(author_id, voter_id)].value if (author_id, voter_id) \
          in self.gamma else 0
      rest -= self.lambd[(author_id, voter_id)].value if (author_id, voter_id) \
          in self.lambd else 0
      variance += 1/parameters.var_H.value
      mean += rest/parameters.var_H.value
    variance = 1 / (1/parameters.var_xi.value + variance)
    mean += parameters.b.value.dot(self.xi[author_id].features) / \
        parameters.var_xi.value
    mean *= variance
    return mean, variance

  def get_gamma_mean_and_variance(self, author_voter, votes, parameters):
    """ Calculates mean and variance of probability distribution of [gamma|Rest].
    
        Args:
          author_voter: a pair with author and voter ids, respectively, to index
            gamma.
          votes: list of vote dictionaries whose author and voter are defined in
            author_voter pair.
          parameters: a ParameterCollection object with the current parameters of
            the model.

        Observation:
          If author and voter are not similar, gamma variable between them is not
        defined and, thus, not considered.

        Returns:
          A pair with the calculated mean and variance.
    """
    variance = 0
    mean = 0
    for vote in votes:
      truth = vote['vote']
      review_id = vote['review']
      author_id, voter_id = author_voter
      if (author_id, voter_id) not in self.gamma:
        continue
      rest = truth - self.alpha[voter_id].value - self.beta[review_id].value - \
          self.xi[author_id].value - \
          self.u[voter_id].value.T.dot(self.v[review_id].value)
      rest -= self.lambd[(author_id, voter_id)].value if (author_id, voter_id) \
          in self.lambd else 0
      variance += 1/parameters.var_H.value
      mean += rest/parameters.var_H.value
    variance = 1 / (1/parameters.var_gamma.value + variance)
    mean += logistic.pdf(parameters.r.value.dot(self.gamma[author_voter]
        .features)) / parameters.var_gamma.value
    mean *= variance
    return mean, variance

  def get_lambda_mean_and_variance(self, author_voter, votes, parameters):
    """ Calculates mean and variance of probability distribution of [lambda|Rest].
    
        Args:
          author_voter: a pair with author and voter ids, respectively, to index
            lambda.
          votes: list of vote dictionaries whose author is author_id.
          parameters: a ParameterCollection object with the current parameters of
            the model.

        Observation:
          If author and voter are not strongly connected, lambda variable between
        them is not defined and, thus, not considered.

        Returns:
          A pair with the calculated mean and variance.
    """
    variance = 0
    mean = 0
    for vote in votes:
      truth = vote['vote']
      review_id = vote['review']
      author_id, voter_id = author_voter
      if (author_id, voter_id) not in self.lambd:
        continue
      rest = truth - self.alpha[voter_id].value - self.beta[review_id].value - \
          self.xi[author_id].value - \
          self.u[voter_id].value.T.dot(self.v[review_id].value)
      rest -= self.gamma[(author_id, voter_id)].value if (author_id, voter_id) \
          in self.gamma else 0
      variance += 1/parameters.var_H.value
      mean += rest/parameters.var_H.value
    variance = 1 / (1/parameters.var_lambda.value + variance)
    mean += logistic.pdf(parameters.h.value.dot(self.lambd[author_voter]
        .features)) / parameters.var_lambda.value
    mean *= variance
    return mean, variance

  def get_u_mean_and_variance(self, voter_id, votes, parameters):
    """ Calculates mean and variance of probability distribution of [u|Rest].
    
        Args:
          voter_id: the id of voter to index u.
          votes: list of vote dictionaries whose voter is voter_id.
          parameters: a ParameterCollection object with the current parameters of
            the model.

        Returns:
          A pair with the calculated mean and variance, with sizes (K, 1) and (K,
        K), respectively.
    """
    variance = np.zeros((constants.K, constants.K))
    mean = np.zeros((constants.K, 1))
    for vote in votes:
      truth = vote['vote']
      review_id = vote['review']
      author_id = vote['reviewer']
      rest = truth - self.alpha[voter_id].value - self.beta[review_id].value
      rest -= self.gamma[(author_id, voter_id)].value if (author_id, voter_id) \
          in self.gamma else 0
      rest -= self.lambd[(author_id, voter_id)].value if (author_id, voter_id) \
          in self.lambd else 0
      variance += self.v[review_id].value.dot(self.v[review_id].value.T) / \
          parameters.var_H.value
      mean += rest * self.v[review_id].value / parameters.var_H.value
    var_u_inv = np.linalg.pinv(parameters.var_u.value)
    variance = np.linalg.pinv(var_u_inv + variance)
    mean = variance.dot(var_u_inv.dot(parameters.W.value.dot(self.u[voter_id]
        .features)) + mean)
    return mean, variance

  def get_v_mean_and_variance(self, review_id, votes, parameters):
    """ Calculates mean and variance of probability distribution of [v|Rest].
    
        Args:
          review_id: the id of review to index v.
          votes: list of vote dictionaries associated with review_id.
          parameters: a ParameterCollection object with the current parameters of
            the model.

        Returns:
          A pair with the calculated mean and variance, with sizes (K, 1) and (K,
        K), respectively.
    """
    variance = np.zeros((constants.K, constants.K))
    mean = np.zeros((constants.K, 1))
    for vote in votes:
      truth = vote['vote']
      voter_id = vote['voter']
      author_id = vote['reviewer']
      rest = truth - self.alpha[voter_id].value - self.beta[review_id].value
      rest -= self.gamma[(author_id, voter_id)].value if (author_id, voter_id) \
          in self.gamma else 0
      rest -= self.lambd[(author_id, voter_id)].value if (author_id, voter_id) \
          in self.lambd else 0
      variance += self.u[voter_id].value.dot(self.u[voter_id].value.T) / \
          parameters.var_H.value
      mean += rest * self.u[voter_id].value / parameters.var_H.value 
    var_v_inv = np.linalg.pinv(parameters.var_v.value)
    variance = np.linalg.pinv(var_v_inv + variance)
    mean = variance.dot(var_v_inv.dot(parameters.V.value.dot(self.v[review_id]
        .features)) + mean)
    return mean, variance

  def calculate_empiric_mean_and_variance(self):
    """ Calculates empiric mean and variance of the variables from samples.

        Args:
          None.

        Returns:
          None. The values of mean and variance are updated on Variable object.
    """
    for variable_group in [self.beta, self.alpha, self.xi, self.gamma,
        self.lambd]:
      for e_id, variable in variable_group.items():
        variable.mean = np.mean(variable.samples)
        variable.var = np.var(variable.samples)
        print variable.name
        print variable.mean
        print variable.var
        print ''
    for variable_group in [self.u, self.v]:
      for e_id, variable in variable_group.items():
        variable.mean = np.mean(variable.samples, axis=0)
        sample = np.array(variable.samples)
        np.reshape(sample, (len(sample), len(sample[0])))
        variable.var = np.cov(sample.T)
            # each variable should be a row
        print variable.name
        print variable.mean
        print variable.var
        print ''


