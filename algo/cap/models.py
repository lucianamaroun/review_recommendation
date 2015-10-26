""" Models Module
    -------------

    Defines variables, parameters and group of variable classes, which are
    fitted using EM algorithm and used for prediction in CAP.

    Not directly callable.
"""


from numpy import array, reshape, mean, std, identity, zeros, ones
from numpy.random import uniform
from scipy.linalg import pinv2 as pinv

from algo.cap import const
from algo.cap.newton_raphson import newton_raphson
from util.aux import sigmoid, sigmoid_der1, sigmoid_der2


class Value(object):
  """ Class representing a value in the algorithm, either a scalar or a matrix.
  """

  def __init__(self, name, shape):
    """ Constructor of value. Any value has a shape and a value, which is 
        randomly initialized.

        Args:
          name: representative name of this value.
          shape: a positive integer or a 2-tuple of positive integers specifying
            the shape of a scalar (in which case it has to be one), a vector or
            a matrix.

        Returns:
          None.
    """
    self.name = name
    self.shape = shape
    self._initialize()

  def _initialize(self):
    """ Initilizes the value according to the shape. The value can be a scalar, 
        in which case the shape is not a tuple, or a matrix. The added values
        are random.

        Observations:
        - Values are generated in a small range of small values (1e-10, 1e-6) 
        for increased change of convergence. 
        - If the shape is not a single positive integer nor a tuple of
        positive integers, an exception is raised and the execution is
        interrupted.
        
        Args:
          None.

        Returns:
          None.
    """
    if type(self.shape) is int and self.shape == 1:
      self.value = uniform(1e-10, 1e-8) # variances cannot be zero
    elif type(self.shape) is tuple and len(self.shape) == 2 and \
        type(self.shape[0]) is int and type(self.shape[1]) is int and \
        self.shape[0] > 0 and self.shape[1] > 0:
      self.value = array([[uniform(1e-10, 1e-8) for _ in
          range(self.shape[1])] for _ in range(self.shape[0])])
    else:
      raise TypeError('TypeError: shape should be a positive int or a 2-tuple of positive ints.')

  def update(self, value):
    """ Updates the value.
        
        Observations: 
        - If the shape of the new value is not the same as the previous one, an
        exception is raised and the execution is interrupted.
       
        Args:
          value: value to replace the old one.
        
        Returns:
          None.
    """
    if (type(value) != type(self.value)):
      raise TypeError('TypeError: value should have type %s, not %s' % 
          (type(self.value), type(value)))
    elif (hasattr(value, 'shape') and self.shape != value.shape):
      raise TypeError('TypeError: value should have shape %s, not %s' 
          % (str(self.shape), str(value.shape)))
    self.value = value 


class Parameter(Value):
  """ Class specifying a Parameter, which helps defining latent variables 
      distributions (regression weights or variances).  
  """

  def __init__(self, name, shape):
    """ Constructor of Parameter.

        Args:
          name: string with the name of the parameter.
          size: integer or pair of matrix dimensions.

        Returns:
          None.
    """
    super(Parameter, self).__init__(name, shape)


class ScalarVarianceParameter(Parameter):
  """ Class specifying a Scalar Parameter representing a distribution's
      variance of a scalar variable.  
  """

  def __init__(self, name):
    """ Constructor of ScalarVarianceParameter. 

        Args:
          name: string with the name of the parameter.

        Returns:
          None.
    """
    super(ScalarVarianceParameter, self).__init__(name, 1)
  
  def optimize(self, variable_group):
    """ Optimizes the value of the parameter using the scalar latent variables
        associated with it and the weight parameter.
    
        Args:
          variable_group: variable group whose variance of distribution is
            represented by this parameter.

        Returns:
          None, but the value field of the parameter is updated.
    """
    size = variable_group.get_size()
    if size > 0:
      w_value = variable_group.weight_param.value
      sse = 0
      var_sum = 0
      for v in variable_group.iter_variables():
        reg = w_value.T.dot(v.features)[0,0]
        sse += (v.empiric_mean - reg)**2
        var_sum += v.empiric_var
      self.update((float) (sse + var_sum) / size)


class PredictionVarianceParameter(ScalarVarianceParameter):
  """ Class specifying the Prediction Parameter, the variance of the true values
      of votes considering the predicted values as means.
  """

  def __init__(self, name):
    """ Constructor of PredictionVarianceParameter. 

        Args:
          name: string with the name of the parameter.

        Returns:
          None.
    """
    super(PredictionVarianceParameter, self).__init__(name)
  
  def optimize(self, groups, votes):
    """ Optimizes the value of the parameter using the prediction and truth
        values.
   
        Observations: 
        - The variance is calculated using RSS (residual sum of squares) of
          predicted and true values of helpfulness votes and empirical variance
          of predicted values.

        Args:
          groups: list of variable group which determines the predicted value.
          votes: list of votes in training set.
           
        Returns:
          None, but the value field of the parameter is updated.
    """
    size = len(votes) 
    sse = 0 # same as rss
    var_sum = 0
    for vote in votes:
      truth = vote['vote']
      pred = 0
      num_samples = groups.values()[0].get_num_samples()
      pred_samples = [0.0] * num_samples
      for g in groups.itervalues():
        inst = g.get_instance(vote)
        if g.pair_name:
          if g.name > g.pair_name:
            continue
          pair_inst = groups[g.pair_name].get_instance(vote)
          pred += inst.empiric_mean.T.dot(pair_inst.empiric_mean)[0,0]
          for i in xrange(num_samples):
            pred_samples[i] += inst.samples[i].T.dot(pair_inst.samples[i])[0,0]
          # V(XY) = V(X)V(Y) + V(X)E(Y)^2 + V(Y)E(X)^2
          # dot product is the sum of variable products
          # dimensions are considered to be uncorrelated
         # dot_var = 0
         # for i in xrange(const.K):
         #   var_x = inst.empiric_var.reshape(-1).tolist()[i]
         #   var_y = pair_inst.empiric_var.reshape(-1).tolist()[i]
         #   dot_var += var_x * var_y
         #   dot_var += var_x * pair_inst.empiric_mean.reshape(-1).tolist()[i]**2
         #   dot_var += var_y * inst.empiric_mean.reshape(-1).tolist()[i]**2
         # var_sum += dot_var
        elif inst:
          pred += inst.empiric_mean
          for i in xrange(num_samples):
            pred_samples[i] += inst.samples[i]
         # var_sum += inst.empiric_var
      var_sum += std(pred_samples, ddof=1) ** 2
      sse += (truth - pred) ** 2
    self.update((float) (sse + var_sum) / size)


class ArrayVarianceParameter(Parameter):
  """ Class specifying a Parameter representing a distribution's variance of an
      Array variable. A single value is used and the covariance is obtained by
      multiplying by the identity.
  """

  def __init__(self, name):
    """ Constructor of ArrayVarianceParameter.
    
        Args:
          name: string with the name of the parameter.

        Returns:
          None.
    """
    super(ArrayVarianceParameter, self).__init__(name, 1)
  
  def optimize(self, variable_group):
    """ Optimizes the value of the parameter using the vector latent variables
        associated with it and the weight parameter.
    
        Observations:
        - As the regression objective is a vector, there are in fact multiple
          regressions. However, there is only one variance parameter. 

        Args:
          variable_group: variable group whose variance of distribution is
            represented by this parameter.

        Returns:
          None, but the value field of the parameter is updated.
    """
    size = variable_group.get_size() * variable_group.get_shape()[0]
    w_value = variable_group.weight_param.value
    sse = 0
    var_sum = 0
    for v in variable_group.iter_variables():
      reg = w_value.dot(v.features)
      sse += sum(((v.empiric_mean - reg)**2).reshape(-1).tolist())
      var_sum += sum(v.empiric_var.reshape(-1).tolist())
    self.update(float(sse + var_sum) / size)


class EntityScalarParameter(Parameter):
  """ Class specifying a  Parameter that is a collection of regression
      weights associated to an Entity Scalar Variable.
  """

  def __init__(self, name, shape):
    """ Constructor of EntityScalarParameter. 

        Args:
          name: string with the name of the parameter.
          shape: pair with vector or matrix dimensions.

        Returns:
          None.
    """
    super(EntityScalarParameter, self).__init__(name, shape)
  
  def optimize(self, variable_group):
    """ Optimizes the value of the weight parameter using the vector latent
        variables associated with it.
    
        Observations:
        - This optimization is the OLS for linear regression solution.
         
        Args:
          variable_group: variable group whose weight of the regression used for
            calculating the mean of the distribution is represented by this
            parameter.

        Returns:
          None, but the value field of the parameter is updated.
    """
    X = array([reshape(v.features, v.features.shape[0]) for v in
        variable_group.iter_variables()])
    y = array([v.empiric_mean for v in variable_group.iter_variables()])
    y = reshape(y, (y.shape[0], 1))
    new_value = pinv(const.ETA * identity(X.shape[1]) + 
        X.T.dot(X)).dot(X.T.dot(y))
    self.update(new_value)


class EntityArrayParameter(Parameter):
  """ Class specifying a  Parameter, that is, collection of regression
      weights, and associated to an Entitiy Array Variable.
  """

  def __init__(self, name, shape):
    """ Constructor of EntityArrayParameter.
     
        Args:
          name: string with the name of the parameter.
          shape pair with matrix dimensions.

        Returns:
          None.
    """
    super(EntityArrayParameter, self).__init__(name, shape)
  
  def optimize(self, variable_group):
    """ Optimizes the value of the weight parameter using the vector latent
        variables associated with it.
    
        Observations:
        - This optimization is the OLS for linear regression solution.
        - Since the regression output is a vector, it is in fact a collection of
        regression problems.
        - We solve the transpose version of the OLS because we want the matrix
        of weights to be (K, f), were K is the number of latent dimensions and
        f the number of features.
        
        Args:
          variable_group: variable group whose weight of the regression used for
            calculating the mean of the distribution is represented by this
            parameter.

        Returns:
          None, but the value field of the parameter is updated.
    """
    X = array([v.features.reshape(-1) for v in variable_group.iter_variables()])
    y = array([v.empiric_mean.reshape(-1) for v in 
        variable_group.iter_variables()])
    new_value = y.T.dot(X).dot(pinv(const.ETA * identity(X.shape[1]) + 
        X.T.dot(X)))
    self.update(new_value)


class InteractionScalarParameter(Parameter):
  """ Class specifying a  Parameter that is a collection of regression
      weights associated to an Interaction Scalar Variable modified by an
      indicator variable.
  """

  def __init__(self, name, shape):
    """ Constructor of InteractionParameter.

        Args:
          name: string with the name of the parameter.
          size: pair with vector or matrix dimensions.

        Returns:
          None.
    """
    super(InteractionScalarParameter, self).__init__(name, shape)
  
  def optimize(self, variable_group):
    """ Optimizes the value of the weight parameter using the indicated scalar
        latent variables associated with it.
    
        Observations:
        - This optimization is OLS, for linear regression solution. Since this
        problem is not a linear regression due to the sigmoid function, 
        the solution used is minimizing the error by finding the value of the
        derivative which is equal to zero. Since the parameter is a vector,
        finding the values of the system of linear equations is not easy and
        it is, thus, approximated by using Newton-Raphson method. 
        
        Args:
          variable_group: variable group whose weight of the regression used for
            calculating the mean of the distribution is represented by this
            parameter.

        Returns:
          None, but the value field of the parameter is updated.
    """
    if variable_group.get_size() > 0:
      new_value = newton_raphson(self.get_derivative_1, self.get_derivative_2, 
          variable_group, self.value)
      self.update(new_value) 

#  def mse(self, value, variable_group):
#    mse = 0.0
#    for variable in variable_group.iter_variables():
#      for sample in variable.samples:
#        mse += (sample - sigmoid(value.T.dot(variable.features)[0,0])) ** 2
#    mse /= variable_group.size * variable.num_samples
#    return mse

  def get_derivative_1(self, value, variable_group):
    """ Gets the first derivative of the expectation with respect to the
        parameter.
    
        Observations:
        - The expectation is the expectation of the log-likelihood with
        respect to the latent variables posterior distribution, found in the
        E-step of the EM method.

        Args:
          value: value of the parameter to calculate the derivative at this
            point.
          variable_group: variable group whose weight of the regression used for
            calculating the mean of the distribution is represented by this
            parameter.
    
        Returns:
          The derivative at point value.
    """
    der = zeros(value.shape)
    for variable in variable_group.iter_variables():
      f = variable.features
      dot = value.T.dot(f)[0,0]
      sig = sigmoid(dot)
      sig1 = sigmoid_der1(dot)
     # for sample in variable.samples:
      der += (min(variable.value, 1.0) - sig) * sig1 * f
   # num_samples = variable_group.get_num_samples()
    der *= 1.0 / (variable_group.var_param.value)# * num_samples)
    return der

  def get_derivative_2(self, value, variable_group):
    """ Gets the second derivative of the expectation with respect to the
        parameter.

        Observations:
        - The expectation is the expectation of the log-likelihood with
        respect to the latent variables posterior distribution, found in the
        E-step of the EM method.

        Args:
          value: value of the parameter to calculate the derivative at this
            point.
          variable_group: variable group whose weight of the regression used for
            calculating the mean of the distribution is represented by this
            parameter.
    
        Returns:
          The derivative at point value.
    """
    der = zeros((value.shape[0], value.shape[0]))
    for variable in variable_group.iter_variables():
      f = variable.features
      dot = value.T.dot(f)[0,0]
      sig = sigmoid(dot)
      sig1 = sigmoid_der1(dot)
      sig2 = sigmoid_der2(dot)
      if variable.feat_matrix is None:
        variable.feat_matrix = f.dot(f.T)
     # for sample in variable.samples:
      der += ((min(variable.value, 1.0) - sig) * sig2 - sig1 * sig1) * variable.feat_matrix 
   # num_samples = variable_group.get_num_samples()
    der *= 1.0 / (variable_group.var_param.value)# * num_samples)
    return der 


class Variable(Value):
  """ Class defining a latent variable.
  """
  
  def __init__(self, name, shape, entity_id, e_type, features, votes): 
    """ Constructor of Variable.
    
        Args:
          name: a string with the variable group name.
          shape: a tuple of positive integers or a positive integer with the
            shape of the variable.
          entity_id: a string with the id of the entity holding this variable.
          e_type: a string or a 2-tuple of strings with entities types related
            to this variable.
          features: array of observed features associated with the variable.
          
        Returns:
          None.
    """
    super(Variable, self).__init__(name, shape)
    if type(self.shape) is int:
      self.value = 0.0 
    else:
      self.value = zeros(self.shape)
    self.entity_id = entity_id
    self.e_type = e_type
    self.features = reshape(features, (features.shape[0], 1))
    self.feat_matrix = None # on demand 
    self.samples = []
    self.num_samples = 0
    self.empiric_mean = None
    self.empiric_var = None
    self.cond_var = None
    self.var_dot = None
    self.related_votes = self.get_related_votes(votes)
    self.num_votes = len(self.related_votes) 

  def add_sample(self, value):
    """ Add a sample of this variable to the list of samples.

        Observations:
        - The sample is used for Gibbs Sampling when approximating the joint
        distribution of latent and observed variables.

        Args:
          value: a value of the type of the variable.
          
        Returns:
          None. The value is added to the list of samples    
    """
    self.num_samples += 1
    self.samples.append(value)

  def get_last_sample(self):
    """ Gets the last sampled value of this variable.

        Args:
          None.

        Returns:
          A value of this variable, with its type.
    """
    if not self.num_samples:
      return self.value # initial value
    return self.samples[-1]

  def get_rest_value(self, groups, vote):
    """ Gets the value of the variable as the truth minus all the other terms
        except the one involving this variable, for a given vote.

        Args:
          groups: dictionary of groups.
          vote: dictionary of a modeled vote defining truth and variable values.

        Returns:
          The value of rest, which is equal to the truth value minus all other
        terms.
    """    
    rest = vote['vote']
    self_group = groups[self.name]
    for group in groups.itervalues():
      if group.name == self.name or group.name == self_group.pair_name:
        continue
      var = group.get_instance(vote)
      if not var:
        continue
      var_value = var.get_last_sample()
      pair_name = group.pair_name
      if pair_name:
        if group.name > pair_name:
          continue
        pair_group = groups[pair_name] 
        pair_value = pair_group.get_instance(vote).get_last_sample()
        rest = rest - var_value.T.dot(pair_value)[0,0]
      else:
        rest = rest - var_value
    return rest

  def reset_samples(self):
    """ Resets sample information after one iteration of EM.

        Args:
          None.

        Returns:
          None. The object is modified with initial values for sample related
        fields.
    """ 
    self.num_samples = 0
    self.samples = []
    self.cond_var = None
    self.var_dot = None
  
#  def generate_dict(self, votes):
#    """ Generates a lightweight dictionary version of this Variable.
#
#        Observation:
#          - Only fields needed for Gibbs Sampling are used.
#
#        Args:
#          votes: list of votes, indexed by id.
#
#        Returns:
#          A dictionary with certain features a Variable object.
#    """
#    d_var = {}
#    d_var['related_votes'] = self.get_related_votes(votes)
#    d_var['num_votes'] = len(d_var['related_votes'])
#    d_var['type'] = self.get_type() 
#    if d_var['type'] == 'EntityArray':
#      d_var['last_matrix'] = None
#    return d_var

  def get_related_votes(self, votes):
    """ Gets related votes' indices of a given Variable.
        
        Args:
          votes: list of votes.

        Returns:
          A list of votes' indices which are associated to this Variable, either
        by review, voter or author (Overriden in InteractionScalarVariable).
    """
    return [i for i, vote in enumerate(votes) if vote[self.e_type] == 
        self.entity_id]


class ScalarVariable(Variable):
  """ Class defining a scalar latent variable.
  """
  
  def __init__(self, name, entity_id, e_type, features, votes): 
    """ Constructor of Variable.
    
        Args:
          name: a string with the name of this variable type.
          entity_id: id of the entity holding this variable.
          e_type: a string with the name of the entity associated to this
            variable (review, author, voter or (author, voter)).
          features: array of observed features associated with the variable.
          votes: list of votes (training set).

        Returns:
          None.
    """
    super(ScalarVariable, self).__init__(name, 1, entity_id, e_type, features,
        votes)
  
  def calculate_empiric_mean(self):
    """ Calculates the empiric mean of this variable using the samples.

        Args:
          None.

        Returns:
          None. The empiric mean field is updated in this object.
    """
    self.empiric_mean = mean(self.samples)
    self.update(float(self.empiric_mean))

  def calculate_empiric_var(self):
    """ Calculates the empiric variance of this variable using the samples.

        Args:
          None.

        Returns:
          None. The empiric variance field is updated in this object.
    """
    self.empiric_var = std(self.samples, ddof=1) ** 2


class ArrayVariable(Variable):
  """ Class defining a latent variable.
  """
  
  def __init__(self, name, shape, entity_id, e_type, features, votes): 
    """ Constructor of Variable.
    
        Args:
          name: string with the Variable name (same as Group).
          shape: a tuple of positive integers or a positive integer with the
            shape of the variable.
          entity_id: id of the entity holding this variable.
          e_type: string with the entity name associated to this variable or a
            tuple of strings if two entities are.
          features: array of observed features associated with the variable.
          votes: list of votes (training set).
          
        Returns:
          None.
    """
    super(ArrayVariable, self).__init__(name, shape, entity_id, e_type,
        features, votes)

  def calculate_empiric_mean(self):
    """ Calculates the empiric mean of this variable using the samples, 
        vector case.

        Args:
          None.

        Returns:
          None. The empiric mean field is updated in this object.
    """
    self.empiric_mean = (1.0 / self.num_samples) * sum(self.samples)
    self.update(self.empiric_mean)

  def calculate_empiric_var(self):
    """ Calculates the empiric variance of this variable using the samples,
        vector case.

        Observation: Covariances are ignored. Then, the result is a list of
          variances, one for each dimension.

        Args:
          None.

        Returns:
          None. The empiric variance field is updated in this object.
    """
    samples = array([s.reshape(-1) for s in self.samples])
    n_dim = samples.shape[1]
    self.empiric_var = reshape(array([[std(samples[:,i], ddof=1) ** 2 for i in
        xrange(n_dim)]]), (n_dim, 1))

  def add_sample(self, value):
    """ Adds a sample to the variable. Differently from other variables, 
        it guarantees the proper shape. 

        Args:
          value: the value of the new sample.

        Returns:
          None.
    """
    value = array(value).reshape(self.shape)
    super(ArrayVariable, self).add_sample(value)


class EntityScalarVariable(ScalarVariable):
  """ Class defining a scalar latent variable associated to an entity.
  """
  
  def __init__(self, name, entity_id, e_type, features, votes): 
    """ Constructor of EntityScalarLatentVariable.
    
        Args:
          name: a string with the name of the latent variable group.
          entity_id: id of the entity holding this variable.
          e_type: a string with the name of the entity associated to this
            variable.
          features: array of observed features associated to the variable.
          votes: list of votes (training set).
          
        Returns:
          None.
    """
    super(EntityScalarVariable, self).__init__(name, entity_id, e_type,
        features, votes)

  def get_cond_mean_and_var(self, groups, votes):
    """ Gets the conditional mean and variance of this variable.
    
        Observations:
        - Returns the mean and variance of this variable used by Gibbs Sampling.
        - The distribution is conditioned on all other latent variables.
        - Some terms of this calculation (cond_var, var_dot) are constant in the
        same EM iteration being, thus, reused and reset in a new iteration
        (through reset_samples function).

        Args:
          groups: a dictionary of Group objects.
          votes: list of votes (training set).

        Returns:
          A 2-tuple with the mean and variance, both floats.
    """
    variance = 0.0
    rest_term = 0.0
    var_group = groups[self.name]
    for i in self.related_votes:
      vote = votes[i] 
      rest = self.get_rest_value(groups, vote)
      rest_term += rest
    rest_term /= var_group.var_H.value 
    if self.cond_var is None:
      self.cond_var = 1.0 / (1.0 / var_group.var_param.value + \
          float(self.num_votes) / var_group.var_H.value)
    if self.var_dot is None:
      self.var_dot = var_group.weight_param.value.T \
          .dot(self.features)[0,0] / var_group.var_param.value
    mean = self.cond_var * (rest_term + self.var_dot)
    return mean, self.cond_var

#  def get_type(self):
#    """ Gets the specific type of this Varible.
#      
#        Args:
#          None.
#
#        Returns:
#          A string with the name of this type.
#    """
#    return 'EntityScalar'


class EntityArrayVariable(ArrayVariable):
  """ Class defining an array latent variable associated to an entity.
  """
  
  def __init__(self, name, shape, entity_id, e_type, features, votes): 
    """ Constructor of Variable.
    
        Args:
          name: a string with the name of this variable.
          shape: a tuple of positive integers or a positive integer with the
            shape of the variable.
          entity_id: id of the entity holding this variable.
          e_type: a string with the name of the entity associated to this
            variable.
          features: array of observed features associated with the variable.
          votes: list of votes (training set).
          
        Returns:
          None.
    """
    super(EntityArrayVariable, self).__init__(name, shape, entity_id, e_type,
        features, votes)
    self.inv_var = None

  def get_cond_mean_and_var(self, groups, votes):
    """ Gets the conditional mean and variance of this variable.
    
        Observations:
        - Returns the mean and variance of this variable used by Gibbs Sampling.
        - The distribution is conditioned on all other latent variables.
        - Some terms of this calculation (inv_var, var_dot) are constant in the
        same EM iteration being, thus, reused and reset in a new iteration
        (through reset_samples function).

        Args:
          groups: a dictionary of Group objects.
          votes: list of votes (training set).

        Returns:
          A 2-tuple with the mean, a vector of size K, and the covariance, a
        matrix of size (K, K).
    """
    variance = zeros((const.K, const.K))
    rest_term = zeros((const.K, 1))
    var_group = groups[self.name]
    pair_group = groups[var_group.pair_name]
    if self.inv_var is None:
      var_matrix = var_group.var_param.value * identity(const.K)
      self.inv_var = pinv(var_matrix)
    for i in self.related_votes:
      vote = votes[i] 
      rest = self.get_rest_value(groups, vote)
      pair_value = pair_group.get_instance(vote).get_last_sample()
      variance += pair_value.dot(pair_value.T)
      rest_term += rest * pair_value
    rest_term /= var_group.var_H.value
    variance = pinv(variance / var_group.var_H.value + self.inv_var)
    if self.var_dot is None:
      w_param = var_group.weight_param.value 
      self.var_dot = self.inv_var.dot(w_param).dot(self.features)
    mean = variance.dot(self.var_dot + rest_term)
    return mean, variance

  def reset_samples(self):
    """ Resets values of fields related to the sampling process, which should be
        initialized between EM iterations.

        Args:
          None.

        Returns:
          None.
    """
    super(EntityArrayVariable, self).reset_samples()
    self.inv_var = None
  
#  def get_type(self):
#    """ Gets the specific type of this Varible.
#      
#        Args:
#          None.
#
#        Returns:
#          A string with the name of this type.
#    """
#    return 'EntityArray'


class InteractionScalarVariable(ScalarVariable):
  """ Class defining a latent variable which is scalar and associated to an
      interaction of entities.
  """
  
  def __init__(self, name, entity_id, e_type, features, votes): 
    """ Constructor of Variable.
    
        Args:
          entity_id: id of the entity holding this variable.
          features: array of observed features associated with the variable.
          votes: list of votes (training set).
          
        Returns:
          None.
    """
    super(InteractionScalarVariable, self).__init__(name, entity_id,
        e_type, features, votes)
 
  def get_cond_mean_and_var(self, groups, votes):
    """ Gets the conditional mean and variance of this variable.
    
        Observations:
        - Returns the variance of this variable used by Gibbs Sampling.
        - The distribution is conditioned on all other latent variables.
        - Some terms of this calculation (cond_var, var_dot) are constant in the
        same EM iteration being, thus, reused and reset in a new iteration
        (through reset_samples function).

        Args:
          groups: a dictionary of Group objects.
          votes: list of votes (training set).

        Returns:
          A 2-tuple with the mean and variance, both float values.
    """
    variance = 0.0
    mean = 0.0
    for i in self.related_votes:
      vote = votes[i] 
      rest = self.get_rest_value(groups, vote)
      mean += rest
    var_group = groups[self.name]
    mean /= var_group.var_H.value
    if self.cond_var is None: 
      self.cond_var = 1.0 / (1.0 / var_group.var_param.value + \
          float(self.num_votes) / var_group.var_H.value)
    if self.var_dot is None:
      self.var_dot = sigmoid(var_group.weight_param.value.T \
          .dot(self.features)[0,0]) / var_group.var_param.value 
    mean = (mean + self.var_dot) * self.cond_var 
    return mean, self.cond_var

  def get_related_votes(self, votes):
    """ Gets related votes associated to an Interaction Variable.

        Args:
          votes: list of votes (training set).

        Returns:
          A list with votes' ids related to both entities of this Interaction
        Variable.
    """
    return [i for i, vote in enumerate(votes) if vote[self.e_type[0]] ==
        self.entity_id[0] and vote[self.e_type[1]] == self.entity_id[1]]
  
#  def get_type(self):
#    """ Gets the specific type of this Varible.
#      
#        Args:
#          None.
#
#        Returns:
#          A string with the name of this type.
#    """
#    return 'InteractionScalar'


class Group(object):
  """ Class container of a set of variables of the same type but regarding 
      different entities.
  """
  
  def __init__(self, name, shape, e_type, weight_param, var_param, var_H):
    """ Initializes the group of variables.

        Args:
          name: a string with the name of the group.
          shape: the shape of each variable instance.
          e_type: entity or tuple relating features' keys (e.g.: review) on 
            votes to entities.
          weight_param: parameter object of regression weights.
          var_param: variance parameter of this variable's distribution.
          var_H: variance parameter of the responde variable, the helpfulness
            vote.
    """
    self.name = name
    self.shape = shape
    self.e_type = e_type
    self.weight_param = weight_param
    self.var_param = var_param
    self.var_H = var_H
    self.variables = {} 
    self.size = 0
    self.pair_name = None

  def iter_variables(self):
    """ Iterates over the instances of variables in this group.
    
        Args:
          None.

        Returns:
          An iterator over Variable objects of this group.
    """
    return self.variables.itervalues()

  def get_instance(self, vote):
    """ Gets an instance variable associated to certain vote.

        Observations:
        - The instance is obtained through the entity value of this variable
        defined in the vote dictionary.

        Args:
          vote: a dictionary of the vote.

        Returns:
          A value of the type of the variable with the sample.
    """
    if type(self.e_type) is tuple:
      e_id = vote[self.e_type[0]], vote[self.e_type[1]]
    else:
      e_id = vote[self.e_type]
    if e_id is None:
      return self.variables[e_id].iter().next()
    if e_id not in self.variables:
      return None
    return self.variables[e_id]
  
  def contains(self, vote):
    """ Checks if the entity of this variable associated with given vote has a
        defined variable, which is true if the indicative value multiplying the
        variable is 1.

        Args:
          vote: the vote with possibly has a variable associated.

        Returns:
          True if the variable is defined or False, otherwise"""
    if type(self.e_type) is tuple:
      e_id = vote[self.e_type[0]], vote[self.e_type[1]]
    else:
      e_id = vote[self.e_type]
    if e_id is None:
      return self.variables[e_id].iter().next()
    if e_id not in self.variables:
      return None
    return e_id in self.variables

  def get_num_samples(self):
    """ Returns the number of samples in the current iteration of EM.

        Args:
          None.

        Returns:
          An interger with the number of samples of each variable.
    """
    return self.iter_variables().next().num_samples

  def get_size(self):
    """ Gets the number of variable instances of this group.

        Args:
          None.

        Returns:
          An integer with the size.
    """
    return self.size 

  def get_shape(self):
    """ Gets the shape of the variables from this group.

        Args:
          None.

        Returns:
          A tuple or int with the shape.
    """
    return self.iter_variables().next().shape

  def set_pair_name(self, pair_name):
    """ Sets pair name of a variable group. A pair name is the name of another
        variable group whose term in the prediction formula is associated to
        this variable, for instance, through a product.

        Args:
          pair_name: a string with the name of the pair variable.

        Returns:
          None. The pair_name field is altered on this object.
    """
    self.pair_name = pair_name

#  def generate_dict(self):
#    """ Generates a lightweight dictionary version of this Group.
#
#        Observation:
#          - Only fields needed for Gibbs Sampling are used.
#
#        Args:
#          votes: list of votes, indexed by id.
#
#        Returns:
#          A dictionary with certain features a Variable object.
#    """
#    d_group = {}
#    d_group['_id'] = str(self.name)
#    d_group['pair_name'] = self.pair_name
#    d_group['entity_type'] = self.e_type
#    d_group['shape'] = self.shape
#    return d_group


class EntityScalarGroup(Group):
  """ Class representing a group of variables which are scalar and
      related to an entity.
  """ 

  def __init__(self, name, e_type, weight_param, var_param, var_H):
    """ Initializes the EntityScalarGroup object.

        Args:
          name: a string with the name of the group.
          e_type: a string with the entity type related to this group.
          weight_param: parameter object of regression weights.
          var_param: variance parameter of this variable's distribution.
          var_H: variance parameter of the response variable (helpfulness vote).

        Returns:
          None.
    """
    super(EntityScalarGroup, self).__init__(name, 1, e_type,
        weight_param, var_param, var_H)

  def add_instance(self, entity_id, features, votes):
    """ Adds an instance variable of this group with the appropriate Variable
        subclass.
    
        Args:
          entity_id: the id of the entity associated to the instance.
          features: an array of features regarding the entity,
          votes: list of votes (training set).
    
        Returns:
          None. The instance is included in the dictionary of instances of this
        object.
    """
    if entity_id in self.variables:
      return 
    self.variables[entity_id] = EntityScalarVariable(self.name, entity_id,
        self.e_type, features, votes)
    self.size += 1


class EntityArrayGroup(Group):
  """ Class representing a group of variables which are vectors and
      entity-related.
  """ 

  def __init__(self, name, shape, e_type, weight_param, var_param, var_H):
    """ Initializes the EntityArrayGroup object. 

        Args:
          name: a string with the name of the group.
          shape: an integer or tuple with the shape of each variable instance.
          e_type: a string with the entity name associated to the variable.
          weight_param: parameter object of regression weights.
          var_param: variance parameter of this variable's distribution.
          var_H: variance parameter of the response variable (helpfulness vote).
        
        Returns:
          None.
    """
    super(EntityArrayGroup, self).__init__(name, shape, e_type, 
        weight_param, var_param, var_H)

  def add_instance(self, entity_id, features, votes): 
    """ Adds an instance variable of this group with the appropriate Variable
        subclass.

        Args:
          entity_id: the id of the entity associated to the instance.
          features: an array of features regarding the entity,
          votes: list of votes (training set).
    
        Returns:
          None. The instance is included in the dictionary of instances of this
        object.
    """ 
    if entity_id in self.variables:
      return 
    self.variables[entity_id] = EntityArrayVariable(self.name, self.shape,
        entity_id, self.e_type, features, votes)
    self.size += 1


class InteractionScalarGroup(Group):
  """ Class representing a set of variables which are scalar and entity-related.
  """ 

  def __init__(self, name, e_type, weight_param, var_param, var_H):
    """ Initializes the InteractionScalarGroup object.

        Args:
          name: a string with the name of the group.
          e_type: a string with the entity type name related to this group.
          weight_param: parameter object of regression weights.
          var_param: variance parameter of this variable's distribution.
          var_H: variance parameter of the response variable (helpfulness vote).
      
        Returns:
          None.
    """
    super(InteractionScalarGroup, self).__init__(name, 1, e_type, weight_param,
        var_param, var_H)

  def add_instance(self, entity_id, features, votes): 
    """ Adds an instance variable of this group with the appropriate Variable
        subclass.

        Args:
          entity_id: the id of the entity associated to the instance.
          features: an array of features regarding the entity,
          votes: list of votes (training set).
    
        Returns:
          None. The instance is included in the dictionary of instances of this
        object.
    """ 
    if entity_id in self.variables:
      return 
    self.variables[entity_id] = InteractionScalarVariable(self.name,
        entity_id, self.e_type, features, votes)
    self.size += 1
