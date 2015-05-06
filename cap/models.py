import numpy as np
from scipy.stats import logistic

import cap.constants as const
from cap.aux import sigmoid, sigmoid_der1, sigmoid_der2
from cap.newton_raphson import newton_raphson

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
    self.value = np.reshape(self.value, size)


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
    self.g = Parameter('g', (, 17))
    self.d = Parameter('d', (1, 9))
    self.b = Parameter('b', (1, 5))
    self.r = Parameter('r', (1, 7))
    self.h = Parameter('h', (1, 4))#5))
    self.W = Parameter('W', (const.K, 9))
    self.V = Parameter('V', (const.K, 17))
    self.var_beta = Parameter('var_beta', 1)
    self.var_alpha = Parameter('var_alpha', 1)
    self.var_xi = Parameter('var_xi', 1)
    self.var_gamma = Parameter('var_gamma', 1)
    self.var_lambda = Parameter('var_lambda', 1)
    self.var_u = Parameter('var_u', 1)
    self.var_v = Parameter('var_v', 1)
    self.var_H = Parameter('var_H', 1)


  def adjust_alpha_related(self, alpha):
    feature_vec = []
    a_vec = []
    for e_id in alpha:
      feature_vec.append(np.reshape(alpha[e_id].features,
          alpha[e_id].features.shape[0]))
      a_vec.append(alpha[e_id].mean)
    feature_vec = np.array(feature_vec)
    a_vec = np.array(a_vec)
    a_vec = np.reshape(a_vec, (a_vec.shape[0], 1))
    inv = np.linalg.pinv(feature_vec.T.dot(feature_vec))
    self.d.value = np.linalg.pinv(feature_vec.T.dot(feature_vec)) \
        .dot(feature_vec.T.dot(a_vec)).T # different from cap (eta not defined),
            # but the same as linear regression optimization
    self.var_alpha.value = sum([(a.mean - self.d.value.dot(a.features))**2 + 
        a.var for a in alpha.values()]) / len(alpha)

  def adjust_beta_related(self, beta):
    feature_vec = []
    b_vec = []
    for e_id in beta:
      feature_vec.append(np.reshape(beta[e_id].features,
          beta[e_id].features.shape[0]))
      b_vec.append(beta[e_id].mean)
    feature_vec = np.array(feature_vec)
    b_vec = np.array(b_vec)
    self.g.value = np.linalg.pinv(feature_vec.T.dot(feature_vec)) \
        .dot(feature_vec.T.dot(b_vec)).T
    self.var_beta.value = sum([(b.mean - self.g.value.dot(b.features))**2 + 
        b.var for b in beta.values()]) / len(beta)

  def adjust_xi_related(self, xi):
    feature_vec = []
    x_vec = []
    for e_id in xi:
      feature_vec.append(np.reshape(xi[e_id].features,
          xi[e_id].features.shape[0]))
      x_vec.append(xi[e_id].mean)
    feature_vec = np.array(feature_vec)
    x_vec = np.array(x_vec)
    self.b.value = np.linalg.pinv(feature_vec.T.dot(feature_vec)) \
        .dot(feature_vec.T.dot(x_vec)).T
    self.var_xi.value = sum([(x.mean - self.b.value.dot(x.features))**2 + 
        x.var for x in xi.values()]) / len(xi)

  def adjust_u_related(self, u):
    feature_vec = []
    u_vec = []
    for e_id in u:
      feature_vec.append(np.reshape(u[e_id].features,
          u[e_id].features.shape[0])) # each user, a row
      u_vec.append(np.reshape(u[e_id].mean, u[e_id].mean.shape[0])) # user row
    feature_vec = np.array(feature_vec)
    u_vec = np.array(u_vec)
    print feature_vec.size
    inv = np.linalg.pinv(feature_vec.T.dot(feature_vec))
    self.W.value = u_vec.T.dot(feature_vec).dot(inv)
    self.var_u.value = 0
    for e in u.values(): # inferrence for u and v, not sure
      diff = e.mean - self.W.value.dot(e.features)
      self.var_u.value += diff.T.dot(diff) + const.K * e.var
    self.var_u.value /= len(u)

  def adjust_v_related(self, v):
    feature_vec = []
    v_vec = []
    for e_id in v:
      feature_vec.append(np.reshape(v[e_id].features,
          v[e_id].features.shape[0])) # each user, a row
      v_vec.append(np.reshape(v[e_id].mean, v[e_id].mean.shape[0])) # user row
    feature_vec = np.array(feature_vec)
    v_vec = np.array(v_vec)
    inv = np.linalg.pinv(feature_vec.T.dot(feature_vec))
    self.V.value = v_vec.T.dot(feature_vec).dot(inv)
    self.var_v.value = 0
    for e in v.values(): # inferrence for u and v, not sure
      diff = e.mean - self.V.value.dot(e.features)
      self.var_v.value += diff.T.dot(diff) + const.K * e.var
    self.var_v.value /= len(v)

  def adjust_gamma_related(self, variables, gamma):
    size = gamma.itervalues().next().features.shape[0]
    r = np.random.random((1, size))
    r = newton_raphson(variables.get_first_derivative_r, 
        variables.get_second_derivative_r, self, r)
    self.var_gamma = sum([(e.mean - self.r.dot(e.features)) ** 2 + e.var \
        for e_id, e in gamma])

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
    self.value = np.reshape(value, (self.size, 1)) if self.size > 1 else value 


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
    self.gamma[author_voter] = Variable('gamma', 1, 'r', 'var_gamma',
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
    self.lambd[author_voter] = Variable('lambda', 1, 'h', 'var_lambda',
        features)

  def add_u_variable(self, voter_id, features):
    """ Adds a u lantent vector associated with a voter.
    
        Args:
          voter_id: string with the id of the voter.
          features: a np.array with size (9, 1) with voter individual features.
        
        Returns:
          None.
    """
    self.u[voter_id] = Variable('u', const.K, 'W', 'var_u', features)


  def add_v_variable(self, review_id, features):
    """ Adds a v lantent vector associated with a review.
    
        Args:
          review_id: string with the id of the review.
          features: a np.array with size (17, 1) with review content features.
        
        Returns:
          None.
    """
    self.v[review_id] = Variable('v', const.K, 'V', 'var_u', features)

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
    variance = np.zeros((const.K, const.K))
    mean = np.zeros((const.K, 1))
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
    matrix_var_u = parameters.var_u.value * np.identity(const.K)
    var_u_inv = np.linalg.pinv(matrix_var_u)
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
    variance = np.zeros((const.K, const.K))
    mean = np.zeros((const.K, 1))
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
    matrix_var_v = parameters.var_v.value * np.identity(const.K)
    var_v_inv = np.linalg.pinv(matrix_var_v)
    variance = np.linalg.pinv(var_v_inv + variance)
    mean = variance.dot(var_v_inv.dot(parameters.V.value.dot(self.v[review_id]
        .features)) + mean)
    return mean, variance

  def get_first_derivative_r(self, r, parameters):
    size = len(self.gamma[self.gamma.iterkeys().next()].features)
    der = [0] * size 
    for author_voter in self.gamma:
      p = self.gamma[author_voter].features
      rp = r.T.dot(p)
      for sample in self.gamma[author_voter].samples:
        der = der + (sigmoid(rp) - self.gamma[author_voter].sample) * \
            sigmoid_der1(rp) * p
    der = 1 / parameters.var_gamma * der
    return np.array(der)    

  def get_second_derivative_r(self, r, parameters):
    size = len(self.gamma[self.gamma.iterkeys().next()].features)
    der = [[0] * size] * size 
    for author_voter in self.gamma:
      p = self.gamma[author_voter].features
      rp = r.dot(p)
      p = p.reshape(1, p.size)
      for sample in self.gamma[author_voter].samples:
        der = der + (sigmoid_der1(rp) ** 2 + 
            (sigmoid(rp) - sample) * sigmoid_der2(rp)) * p.T.dot(p)
    der = 1 / parameters.var_gamma * der
    return der    


