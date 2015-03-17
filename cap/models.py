import numpy as np
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
    self.value = np.zeros(size)
    for i in range(size[0]):
      for j in range(size[1]):
        self.value[i,j] = np.random.uniform(0, 1)


class ParameterCollection(object):
  """ Class specifying the collection of parameters of the model. Each parameter
      is singularly defined, that is, there is only one for all the instances.
  """

  def __init__(self, name, size):
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
    self.h = Parameter('h', (1, 5))
    self.W = Parameter('W', (constants.K, 9))
    self.V = Parameter('V', (constants.K, 17))
    self.var_beta = Parameter('var_beta', (1, 1))
    self.var_alpha = Parameter('var_alpha', (1, 1))
    self.var_xi = Parameter('var_xi', (1, 1))
    self.var_gamma = Parameter('var_gamma', (1, 1))
    self.var_lambda = Parameter('var_lambda', (1, 1))
    self.var_u = Parameter('var_u', (K, K))
    self.var_v = Parameter('var_v', (K, K))
    self.var_H = Parameter('var_H', (1, 1))


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
    self.features = features
    self.value = np.zeros(size) 


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
    self.u[review_id] = Variable('v', constants.K, 'V', 'var_u', features)

  def get_alpha_mean_and_variance(self, voter_id, votes, sim, trust, parameters):
    """ Calculates mean and variance of probability distribution of [alpha|Rest].
    
        Args:
          voter_id: the id of voter to index alpha.
          votes: list of vote dictionaries whose voter was voter_id.
          sim: dictionary indexed by pairs (voter_id, author_id) with the binary
            similarity between author and voter. It is not symmetric: it indicates
            if author_id is in the set of similar users of voter_id.
          trust: dictionary indexed by pairs (voter_id, author_id) with the binary
            value indicating if voter_id trusts author_id.
          parameters: a ParameterCollection object with the current parameters of
            the model.

        Returns:
          A pair with the calculated mean and variance.
    """
    variance = 0
    mean = 0
    for vote in votes:
      review_id = vote['review']
      author_id = vote['author']
      rest = truth - self.beta[review_id].value - self.xi[author_id].value - \
          sim[(voter_id, author_id)] * self.gamma[(author_id, voter_id)].value - \
          trust[(voter_id, author_id)] * self.lamb[(author_id, voter_id)].value - \
          self.u[voter_id].value.T.dot(self.v[review_id].value)
      variance += 1/parameters.var_H + 1/parameters.var_alpha
      mean += rest/parameters.var_H + \
          parameters.d.dot(self.alpha[voter_id].features) / \
          parameters.var_alpha
    variance = 1/variance
    mean *= variance
    return mean, variance

  def get_beta_mean_and_variance(self, review_id, votes, sim, trust, parameters):
    """ Calculates mean and variance of probability distribution of [beta|Rest].
    
        Args:
          review_id: the id of voter to index beta.
          votes: list of vote dictionaries associated with the review.
          sim: dictionary indexed by pairs (voter_id, author_id) with the binary
            similarity between author and voter. It is not symmetric: it indicates
            if author_id is in the set of similar users of voter_id.
          trust: dictionary indexed by pairs (voter_id, author_id) with the binary
            value indicating if voter_id trusts author_id.
          parameters: a ParameterCollection object with the current parameters of
            the model.

        Returns:
          A pair with the calculated mean and variance.
    """
    variance = 0
    mean = 0
    for vote in votes:
      voter_id = vote['voter']
      author_id = vote['author']
      rest = truth - self.alpha[voter_id].value - self.xi[author_id].value - \
          sim[(voter_id, author_id)] * self.gamma[(author_id, voter_id)].value - \
          trust[(voter_id, author_id)] * self.lamb[(author_id, voter_id)].value - \
          self.u[voter_id].value.T.dot(self.v[review_id].value)
      variance += 1/parameters.var_H + 1/parameters.var_beta
      mean += rest/parameters.var_H + \
          parameters.d.dot(self.beta[review_id].features) / \
          parameters.var_beta
    variance = 1/variance
    mean *= variance
    return mean, variance
    
  def get_xi_mean_and_variance(self, author_id, votes, sim, trust, parameters):
    """ Calculates mean and variance of probability distribution of [xi|Rest].
    
        Args:
          author_id: the id of voter to index xi.
          votes: list of vote dictionaries whose author is author_id.
          sim: dictionary indexed by pairs (voter_id, author_id) with the binary
            similarity between author and voter. It is not symmetric: it indicates
            if author_id is in the set of similar users of voter_id.
          trust: dictionary indexed by pairs (voter_id, author_id) with the binary
            value indicating if voter_id trusts author_id.
          parameters: a ParameterCollection object with the current parameters of
            the model.

        Returns:
          A pair with the calculated mean and variance.
    """
    variance = 0
    mean = 0
    for vote in votes:
      review_id = vote['review']
      voter_id = vote['voter']
      rest = truth - self.alpha[voter_id].value - self.beta[review_id].value - \
          sim[(voter_id, author_id)] * self.gamma[(author_id, voter_id)].value - \
          trust[(voter_id, author_id)] * self.lamb[(author_id, voter_id)].value - \
          self.u[voter_id].value.T.dot(self.v[review_id].value)
      variance += 1/parameters.var_H + 1/parameters.var_xi
      mean += rest/parameters.var_H + \
          parameters.d.dot(self.xi[author_id].features)/parameters.var_xi
    variance = 1/variance
    mean *= variance
    return mean, variance

  def get_gamma_mean_and_variance(self, author_voter, votes, sim, trust, 
    parameters):
    """ Calculates mean and variance of probability distribution of [gamma|Rest].
    
        Args:
          author_voter: a pair with author and voter ids, respectively, to index
            gamma.
          votes: list of vote dictionaries whose author and voter are defined in
            author_voter pair.
          sim: dictionary indexed by pairs (voter_id, author_id) with the binary
            similarity between author and voter. It is not symmetric: it indicates
            if author_id is in the set of similar users of voter_id.
          trust: dictionary indexed by pairs (voter_id, author_id) with the binary
            value indicating if voter_id trusts author_id.
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
      review_id = vote['review']
      author_id, voter_id = author_voter
      if not sim[(author_id, voter_id)]:
        continue
      rest = truth - self.alpha[voter_id].value - self.beta[review_id].value - \
          self.xi[author_id].value - \
          trust[(voter_id, author_id)] * self.lamb[(author_id, voter_id)].value - \
          self.u[voter_id].value.T.dot(self.v[review_id].value)
      variance += 1/parameters.var_H + 1/parameters.var_gamma
      mean += rest/parameters.var_H + \
          parameters.d.dot(self.gamma[author_voter].features) / \
          parameters.var_gamma
    variance = 1/variance
    mean *= variance
    return mean, variance

  def get_lambda_mean_and_variance(self, author_voter, votes, sim, trust, 
    parameters):
    """ Calculates mean and variance of probability distribution of [lambda|Rest].
    
        Args:
          author_voter: a pair with author and voter ids, respectively, to index
            lambda.
          votes: list of vote dictionaries whose author is author_id.
          sim: dictionary indexed by pairs (voter_id, author_id) with the binary
            similarity between author and voter. It is not symmetric: it indicates
            if author_id is in the set of similar users of voter_id.
          trust: dictionary indexed by pairs (voter_id, author_id) with the binary
            value indicating if voter_id trusts author_id.
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
      review_id = vote['review']
      author_id, voter_id = author_voter
      if not trust[(author_id, voter_id)]:
        continue
      rest = truth - self.alpha[voter_id].value - self.beta[review_id].value - \
          self.xi[author_id].value - \
          sim[(voter_id, author_id)] * self.gamma[(author_id, voter_id)].value - \
          self.u[voter_id].value.T.dot(self.v[review_id].value)
      variance += 1/parameters.var_H + 1/parameters.var_lambda
      mean += rest/parameters.var_H + \
          parameters.d.dot(self.lambd[author_voter])/parameters.var_lambda
    variance = 1/variance
    mean *= variance
    return mean, variance

  def get_u_mean_and_variance(self, voter_id, votes, sim, trust, parameters):
    """ Calculates mean and variance of probability distribution of [u|Rest].
    
        Args:
          voter_id: the id of voter to index u.
          votes: list of vote dictionaries whose voter is voter_id.
          sim: dictionary indexed by pairs (voter_id, author_id) with the binary
            similarity between author and voter. It is not symmetric: it indicates
            if author_id is in the set of similar users of voter_id.
          trust: dictionary indexed by pairs (voter_id, author_id) with the binary
            value indicating if voter_id trusts author_id.
          parameters: a ParameterCollection object with the current parameters of
            the model.

        Returns:
          A pair with the calculated mean and variance, with sizes (K, 1) and (K,
        K), respectively.
    """
    variance = np.zeros(constants.K, constants.K)
    mean = np.zeros(constants.K, 1)
    for vote in votes:
      review_id = vote['review']
      author_id = vote['author']
      rest = truth - self.alpha[voter_id] - self.beta[review_id] - \
          sim[(voter_id, author_id)] * self.gamma[(author_id, voter_id)] - \
          trust[(voter_id, author_id)] * self.lamb[(author_id, voter_id)]
      variance += self.v[review_id].dot(self.v[review_id].T) / \
          parameters.var_H
      mean += rest*self.v[review_id]/parameters.var_H
    var_u_inv = np.linalg.inv(parameters.var_u)
    variance = np.linalg.inv(var_u_inv + variance)
    mean = variance * (var_u_inv.dot(parameters.W.dot(features)) + mean)
    return mean, variance

  def get_v_mean_and_variance(self, review_id, votes, sim, trust, parameters):
    """ Calculates mean and variance of probability distribution of [v|Rest].
    
        Args:
          review_id: the id of review to index v.
          votes: list of vote dictionaries associated with review_id.
          sim: dictionary indexed by pairs (voter_id, author_id) with the binary
            similarity between author and voter. It is not symmetric: it indicates
            if author_id is in the set of similar users of voter_id.
          trust: dictionary indexed by pairs (voter_id, author_id) with the binary
            value indicating if voter_id trusts author_id.
          parameters: a ParameterCollection object with the current parameters of
            the model.

        Returns:
          A pair with the calculated mean and variance, with sizes (K, 1) and (K,
        K), respectively.
    """
    variance = np.zeros(constants.K, constants.K)
    mean = np.zeros(constants.K, 1)
    for vote in votes:
      voter_id = vote['voter']
      author_id = vote['author']
      rest = truth - self.alpha[voter_id] - self.beta[review_id] - \
          sim[(voter_id, author_id)] * self.gamma[(author_id, voter_id)] - \
          trust[(voter_id, author_id)] * self.lamb[(author_id, voter_id)]
      variance += self.u[voter_id].dot(self.u[voter_id].T) / \
          parameters.var_H
      mean += rest*self.u[voter_id]/parameters.var_H 
    var_v_inv = np.linalg.inv(parameters.var_v)
    variance = np.linalg.inv(var_v_inv + variance)
    mean = variance * (var_v_inv.dot(parameters.D.dot(features)) + mean)
    return mean, variance
