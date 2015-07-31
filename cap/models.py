import numpy as np
from numpy import array, random, reshape, mean, std, identity, zeros, diagonal
from numpy.linalg import pinv

from cap import const
from cap.aux import sigmoid, sigmoid_der1, sigmoid_der2
from cap.newton_raphson import newton_raphson


class Value(object):
  """ Class representing a value in the algorithm, either a scalar or a matrix.
  """

  def __init__(self, name, shape):
    """ Constructor of value. Any value has a shape and a value, which is 
        randomly initialized.

        Args:
          name: representative name of this value.
          shape: a positive integer or a 2-tuple of positive integers specifying
            the shape of a scalar or a matrix, respectively.

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
        - To assure the values are bigger than 1, a small positive value is
        added.
        - If the shape is not a single positive integer nor a tuple of
        positive integers, an exception is raised and the execution is
        interrupted.
        
        Args:
          None.

        Returns:
          None.
    """
    if type(self.shape) is int and self.shape == 1:
      self.value = random.random() + 0.0000001
    elif type(self.shape) is tuple and len(self.shape) == 2 and \
        type(self.shape[0]) is int and type(self.shape[1]) is int and \
        self.shape[0] > 0 and self.shape[1] > 0:
      self.value = array([[random.random() + 0.0000001 for _ in
          range(self.shape[1])] for _ in range(self.shape[0])])
    else:
      raise TypeError('TypeError: shape should be a positive int or a 2-tuple of positive ints.')

  def update(self, value):
    """ Updates the value.
        
        Observations: 
        - If the shape is the new value is not the same as the previous one, an
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
      raise TypeError('TypeError: value should have shape (%d, %d), not (%d, %d)' 
          % (self.shape[0], self.shape[1], value.shape[0], value.shape[1]))
    self.value = value 


class Parameter(Value):
  """ Class specifying a Parameter, which defines latent variables distributions
      and relationship with observed variables.  
  """

  def __init__(self, name, shape):
    """ Constructor of Parameter.

        Args:
          name: string with the name of the parameter.
          size: pair with vector or matrix dimensions.

        Returns:
          None.
    """
    super(Parameter, self).__init__(name, shape)


class ScalarVarianceParameter(Parameter):
  """ Class specifying a Scalar Parameter representing a distribution's
      variance.  
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
            represented by this paramter.

        Returns:
          None, but the value field of the parameter is updated.
    """
    size = variable_group.get_size()
    w_value = variable_group.weight_param.value
    sse = 0
    var_sum = 0
    for v in variable_group.iter_variables():
      reg = w_value.T.dot(v.features)[0,0]
      sse += (v.empiric_mean - reg)**2
      var_sum += v.empiric_var
    self.update((float) (sse + var_sum) / size)


class PredictionVarianceParameter(ScalarVarianceParameter):
  """ Class specifying the Prediction Parameter representing the estimated value
      distribution's variance.  
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
   
        Observation: The latent variables are considered to be uncorrelated,
          implying that the prediction empiric variance is the sum of latent
          variables empiric variances.
    
        Args:
          gropus: list of variable group which determines the predicted value.
           
        Returns:
          None, but the value field of the parameter is updated.
    """
    size = groups.itervalues().next().get_size()
    sse = 0
    var_sum = 0
    for vote in votes:
      truth = vote['vote']
      pred = 0
      for g in groups.itervalues():
        inst = g.get_instance(vote)
        if g.pair_name:
          if g.name > g.pair_name:
            continue
          pair_inst = groups[g.pair_name].get_instance(vote)
          pred += inst.empiric_mean.T.dot(pair_inst.empiric_mean)[0,0]
          # V(XY) = V(X)V(Y) + V(X)E(Y)^2 + V(Y)E(X)^2
          # dot product is the sum of variable products
          # dimensions are considered to be uncorrelated
          prod_var = 0
          for i in xrange(const.K):
            var_x = inst.empiric_var.reshape(-1).tolist()[i]
            var_y = pair_inst.empiric_var.reshape(-1).tolist()[i]
            prod_var += var_x * var_y
            prod_var += var_x * pair_inst.empiric_mean.reshape(-1).tolist()[i]
            prod_var += var_y * inst.empiric_mean.reshape(-1).tolist()[i]
          var_sum += prod_var
        elif inst:
          pred += inst.empiric_mean
          var_sum += inst.empiric_var
        sse += (truth - pred)**2
    self.update((float) (sse + var_sum) / size)


class ArrayVarianceParameter(Parameter):
  """ Class specifying a Parameter representing a distribution's
      variance of an Array variable. A single value is used and the covariance
      is obtained by multiplying by de identity.
  """

  def __init__(self, name):
    """ Constructor of ArrayVarianceParameter.
    
        Args:
          name: string with the name of the parameter.
          size: pair with vector or matrix dimensions.

        Returns:
          None.
    """
    super(ArrayVarianceParameter, self).__init__(name, 1)
  
  def optimize(self, variable_group):
    """ Optimizes the value of the parameter using the vector latent variables
        associated with it and the weight parameter.
    
        Observations:
        - As the regression objective is a vector, there are in fact multiple
          regressions. However, there is only one variance parameter. This, the
          size if multiplied by the dimension of the variable (although it is
          not clear in the papers Regression-based Latent Factor Models KDD'09
          nor in Context-aware Review Helpfulness Rating Prediction RecSys'13.

        Args:
          variable_group: variable group whose variance of distribution is
            represented by this paramter.

        Returns:
          None, but the value field of the parameter is updated.
    """
    size = variable_group.get_size() * variable_group.iter_variables().next().\
        value.shape[0]
    w_value = variable_group.weight_param.value
    sse = 0
    var_sum = 0
    for v in variable_group.iter_variables():
      reg = w_value.dot(v.features)
      sse += sum(((v.empiric_mean - reg)**2).reshape(-1).tolist())
      var_sum += sum(diagonal(v.empiric_var).tolist())
    self.update((sse + var_sum) / size)


class EntityScalarParameter(Parameter):
  """ Class specifying a  Parameter, that is, collection of regression
      weights, and associated to an Entity Scalar Variable.
  """

  def __init__(self, name, shape):
    """ Constructor of EntityScalarParameter. 

        Args:
          name: string with the name of the parameter.
          size: pair with vector or matrix dimensions.

        Returns:
          None.
    """
    super(EntityScalarParameter, self).__init__(name, shape)
  
  def optimize(self, variable_group):
    """ Optimizes the value of the weight parameter using the vector latent
        variables associated with it.
    
        Observations:
        - This optimization is the OLS for linear regression solution.
        - The solution is a little bit different from Context-aware Review
        Helpfulness Rating Prediction RecSys'13 because there is an eta variable
        in their solution that is not specified.
         
        Args:
          variable_group: variable group whose weight of the regression used for
            calculating the mean of the distribution is represented by this
            paramter.

        Returns:
          None, but the value field of the parameter is updated.
    """
    feat_matrix = array([reshape(v.features, v.features.shape[0]) 
        for v in variable_group.iter_variables()])
    y = array([v.empiric_mean for v in variable_group.iter_variables()])
    y = reshape(y, (y.shape[0], 1))
    inv = pinv(feat_matrix.T.dot(feat_matrix))
    new_value = pinv(feat_matrix.T.dot(feat_matrix)) \
        .dot(feat_matrix.T.dot(y))
    self.update(new_value)


class EntityArrayParameter(Parameter):
  """ Class specifying a  Parameter, that is, collection of regression
      weights, and associated to an Entitiy Array Variable.
  """

  def __init__(self, name, shape):
    """ Constructor of EntityArrayParameter.
     
        Args:
          name: string with the name of the parameter.
          size: pair with vector or matrix dimensions.

        Returns:
          None.
    """
    super(EntityArrayParameter, self).__init__(name, shape)
  
  def optimize(self, variable_group):
    """ Optimizes the value of the weight parameter using the vector latent
        variables associated with it.
    
        Observations:
        - This optimization is the OLS for linear regression solution.
        - The solution is a little bit different from Context-aware Review
        Helpfulness Rating Prediction RecSys'13 because there is an eta variable
        in their solution that is not specified.
        - Since the regression output is a vector, it is in fact a collection of
        regression problems.     
        
        Args:
          variable_group: variable group whose weight of the regression used for
            calculating the mean of the distribution is represented by this
            paramter.

        Returns:
          None, but the value field of the parameter is updated.
    """
    feat_matrix = array([v.features.reshape(-1) for v in 
        variable_group.iter_variables()])
    y = array([v.empiric_mean.reshape(-1) for v in 
        variable_group.iter_variables()])
    inv = pinv(feat_matrix.T.dot(feat_matrix))
    new_value = y.T.dot(feat_matrix).dot(inv)
    self.update(new_value)


class InteractionScalarParameter(Parameter):
  """ Class specifying a  Parameter, that is, collection of regression
      weights, and associated to an Interaction Scalar Variable modified by an
      indicator variable ().
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
        - This optimization is the OLS for linear regression solution.
        - Since the problem is not a linear regression due to the sigmoid
        function, the problem is converted into minimizing the error by
        finding the value of the derivative which is equal to zero. Since the
        parameter is a vector, finding the values of this derivative is not easy
        and it is, thus, approximated by using Newton-Raphson method. 
        
        Args:
          variable_group: variable group whose weight of the regression used for
            calculating the mean of the distribution is represented by this
            paramter.

        Returns:
          None, but the value field of the parameter is updated.
    """
    new_value = newton_raphson(self.get_derivative_1, self.get_derivative_2, 
        variable_group, self.value)
    self.update(new_value) 

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
            paramter.
    
        Returns:
          The derivative at point value.
    """
    size = variable_group.get_size()
    der = np.zeros(variable_group.weight_param.shape) 
    for variable in variable_group.iter_variables():
      f = variable.features
      dot = value.T.dot(f)[0,0]
      for sample in variable.samples:
        try:
          der = der + (sigmoid(dot) - sample) * sigmoid_der1(dot) * f
        except:
          print dot
          print sigmoid(dot)
          print der
          import sys
          sys.exit()
    der = 1 / variable_group.var_param.value * der
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
    size = variable_group.get_size()
    der = np.zeros((value.shape[0], value.shape[0])) 
    for variable in variable_group.iter_variables():
      f = variable.features
      dot = value.T.dot(f)[0,0]
      for sample in variable.samples:
        try:
          der = der + (sigmoid_der1(dot) ** 2 + (sigmoid(dot) - sample) * \
              sigmoid_der2(dot)) * f.dot(f.T)
        except:
          print "value ",
          print value
          print "f ",
          print f
          print "dot ",
          print dot
          print "sigmoid ",
          print sigmoid(dot)
          print "sigmoid_der1 ",
          print sigmoid_der1(dot)
          print "sigmoid_der2 ",
          print sigmoid_der2(dot)
          print "sample ",
          print sample
          print "f dot ",
          print f.dot(f.T)
          import sys
          sys.exit()
    der = 1 / variable_group.var_param.value * der
    return der 


class Variable(Value):
  """ Class defining a latent variable.
  """
  
  def __init__(self, name, shape, entity_id, e_type, features): 
    """ Constructor of Variable.
    
        Args:
          name: a string with the variable group name.
          shape: a tuple of positive integers or a positive integer with the
            shape of the variable.
          entity_id: a string with the id of the entity holding this variable.
          e_type: a string or a tuple with entities types related to this
            variable.
          features: array of observed features associated with the variable.
          
        Returns:
          None.
    """
    super(Variable, self).__init__(name, shape)
    self.entity_id = entity_id
    self.e_type = e_type
    self.features = reshape(features, (features.shape[0], 1))
    self.samples = []
    self.empiric_mean = None
    self.empiric_var = None

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
    self.samples.append(value)

  def get_last_sample(self):
    """ Gets the last sampled value of this variable.

        Args:
          None.

        Returns:
          A value of this variable, with its type.
    """
    if not self.samples:
      return self.value # initial value
    return self.samples[-1]


  def get_rest_value(self, groups, vote):
    """ Gets the value of the variable as the truth minus all the other terms
        except the one involving this variable, for a given vote.

        Args:
          groups: dictionary of groups.
          vote: dictionary of a modeled vote.

        Returns:
          The value of rest, which is equal to the truth value minus all other
        terms.
    """    
    truth = vote['vote']
    rest = truth
    names = groups.keys()[:]
    self_group = groups[self.name]
    for name in names:
      if name == self.name or name == self_group.pair_name:
        continue
      group = groups[name]
      var = group.get_instance(vote)
      if not var:
        continue
      var_value = var.get_last_sample()
      if group.pair_name:
        pair_name = group.pair_name
        pair_value = groups[pair_name].get_instance(vote).get_last_sample()
        rest = rest - var_value.T.dot(pair_value)[0,0]
        names.remove(pair_name) # pair is not processed again
      else:
        rest = rest - var_value
    return rest

class ScalarVariable(Variable):
  """ Class defining a scalar latent variable.
  """
  
  def __init__(self, name, entity_id, e_type, features): 
    """ Constructor of Variable.
    
        Args:
          name: a string with the name of this variable type.
          entity_id: id of the entity holding this variable.
          e_type: a string with the name of the entity associated to this
            variable (review, reviewer or votes).
          features: array of observed features associated with the variable.
          
        Returns:
          None.
    """
    super(ScalarVariable, self).__init__(name, 1, entity_id, e_type, features)

  def calculate_empiric_mean(self):
    """ Calculates the empiric mean of this variable using the samples.

        Args:
          None.

        Returns:
          None. The empiric mean field is updated in this object.
    """
    self.empiric_mean = mean(self.samples)

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
  
  def __init__(self, name, shape, entity_id, e_type, features): 
    """ Constructor of Variable.
    
        Args:
          shape: a tuple of positive integers or a positive integer with the
            shape of the variable.
          entity_id: id of the entity holding this variable.
          features: array of observed features associated with the variable.
          
        Returns:
          None.
    """
    super(ArrayVariable, self).__init__(name, shape, entity_id, e_type,
        features)

  def calculate_empiric_mean(self):
    """ Calculates the empiric mean of this variable using the samples, 
        vector case.

        Args:
          None.

        Returns:
          None. The empiric mean field is updated in this object.
    """
    self.empiric_mean = (1.0 / len(self.samples)) * sum(self.samples)

  def calculate_empiric_var(self):
    """ Calculates the empiric variance of this variable using the samples,
        vector case.

        Args:
          None.

        Returns:
          None. The empiric variance field is updated in this object.
    """
    samples = array([s.reshape(-1) for s in self.samples])
    n_dim = samples.shape[1]
    self.empiric_var = reshape(array([[std(samples[:,i], ddof=1) ** 2 for i in
        xrange(n_dim)]]), (n_dim, 1))


class EntityScalarVariable(ScalarVariable):
  """ Class defining a scalar latent variable associated to an entity.
  """
  
  def __init__(self, name, entity_id, e_type, features): 
    """ Constructor of EntityScalarLatentVariable.
    
        Args:
          name: a string with the name of the latent variable group.
          entity_id: id of the entity holding this variable.
          e_type: a string with the name of the entity associated to this
            variable.
          features: array of observed features associated with the variable.
          
        Returns:
          None.
    """
    super(EntityScalarVariable, self).__init__(name, entity_id, e_type,
        features)

  def get_cond_mean_and_var(self, groups, votes):
    """ Calculates the conditional mean and variance of this variable.
    
        Observations:
        - Returns the mean and variance of this variable used by Gibbs Sampling.
        - The distribution is conditioned on all other latent variables.

        Args:
          groups: a dictionary of Group objects.
          votes: the list of votes (training set).

        Returns:
          A 2-tuple with the mean and variance, both floats.
    """
    related_votes = [v for v in votes if v[self.e_type] == self.entity_id]
    variance = 0
    mean = 0
    var_group = groups[self.name]
    for vote in related_votes:
      rest = self.get_rest_value(groups, vote)
      variance += 1
      mean += rest
    variance /= var_group.var_H.value
    mean /= var_group.var_H.value
    variance = 1 / (1/var_group.var_param.value + variance)
    dot = var_group.weight_param.value.T.dot(self.features) / \
        var_group.var_param.value
    mean = variance * (mean + dot[0,0])
    return mean, variance


class EntityArrayVariable(ArrayVariable):
  """ Class defining a latent variable.
  """
  
  def __init__(self, name, shape, entity_id, e_type, features): 
    """ Constructor of Variable.
    
        Args:
          shape: a tuple of positive integers or a positive integer with the
            shape of the variable.
          entity_id: id of the entity holding this variable.
          features: array of observed features associated with the variable.
          
        Returns:
          None.
    """
    super(EntityArrayVariable, self).__init__(name, shape, entity_id, e_type,
        features)

  def get_cond_mean_and_var(self, groups, votes):
    """ Calculates the conditional mean and variance of this variable.
    
        Observations:
        - Returns the mean and variance of this variable used by Gibbs Sampling.
        - The distribution is conditioned on all other latent variables.

        Args:
          groups: a dictionary of Group objects.
          votes: the list of votes (training set).

        Returns:
          A 2-tuple with the mean, a vector of size K, and variance, a
        covariance matrix of shape (K, K).
    """
    related_votes = [v for v in votes if v[self.e_type] == self.entity_id]
    variance = zeros((const.K, const.K))
    mean = zeros((const.K, 1))
    var_group = groups[self.name]
    pair_group = groups[var_group.pair_name]
    var_matrix = var_group.var_param.value * identity(const.K)
    inv_var = pinv(var_matrix)
    for vote in related_votes:
      rest = self.get_rest_value(groups, vote)
      pair_value = pair_group.get_instance(vote).get_last_sample()
      variance += pair_value.dot(pair_value.T)
      mean += rest * pair_value
    variance = pinv(variance / var_group.var_H.value + inv_var)
    dot = inv_var.dot(var_group.weight_param.value).dot(self.features)
    mean = variance.dot(dot + mean / var_group.var_H.value)
    return mean, variance


class InteractionScalarVariable(ScalarVariable):
  """ Class defining a latent variable.
  """
  
  def __init__(self, name, entity_id, e_type, features): 
    """ Constructor of Variable.
    
        Args:
          entity_id: id of the entity holding this variable.
          features: array of observed features associated with the variable.
          
        Returns:
          None.
    """
    super(InteractionScalarVariable, self).__init__(name, entity_id,
        e_type, features)
 
  def get_cond_mean_and_var(self, groups, votes):
    """ Calculates the conditional mean and variance of this variable.
    
        Observations:
        - Returns the mean and variance of this variable used by Gibbs Sampling.
        - The distribution is conditioned on all other latent variables.

        Args:
          groups: a dictionary of Group objects.
          votes: the list of votes (training set).

        Returns:
          A 2-tuple with the mean and variance, both float values.
    """
    related_votes = [v for v in votes if v[self.e_type[0]] == self.entity_id[0]
        and v[self.e_type[1]] == self.entity_id[1]]
    variance = 0
    mean = 0
    var_group = groups[self.name]
    for vote in related_votes:
      rest = self.get_rest_value(groups, vote)
      variance += 1/var_group.var_H.value
      mean += rest/var_group.var_H.value
    variance = 1 / (1/var_group.var_param.value + variance)
    mean += sigmoid(var_group.weight_param.value.T.dot(self.features)[0,0]) \
        / var_group.var_param.value
    mean *= variance
    return mean, variance


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
    self.pair_name = None
  
  def iter_variables(self):
    """ Iterates over the instances of this variable.
    
        Args:
          None.

        Returns:
          An iterator over Variable objects of this group.
    """
    for variable in self.variables.itervalues():
      yield variable 

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


  def get_size(self):
    """ Gets the number of instances of this variable.

        Args:
          None.

        Returns:
          An integer with the size.
    """
    return len(self.variables)

  def set_pair_name(self, pair_name):
    """ Sets pair name of a variable. A pair name is the name of another
        variable whose term in the prediction formula is associated to this
        variable, for instance, through a product.

        Args:
          pair_name: a string with the name of the pair variable.

        Returns:
          None. The pair_name field is altered on this object.
    """
    self.pair_name = pair_name


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

  def add_instance(self, entity_id, features):
    """ Adds an instance variable of this group with the appropriate Variable
        subclass.
    
        Args:
          entity_id: the id of the entity associated to the instance.
          features: an array of features regarding the entity,
    
        Returns:
          None. The instance is included in the dictionary of instances of this
        object.
    """
    if entity_id in self.variables:
      return 
    self.variables[entity_id] = EntityScalarVariable(self.name, entity_id,
        self.e_type, features)


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

  def add_instance(self, entity_id, features): 
    """ Adds an instance variable of this group with the appropriate Variable
        subclass.

        Args:
          entity_id: the id of the entity associated to the instance.
          features: an array of features regarding the entity,
    
        Returns:
          None. The instance is included in the dictionary of instances of this
        object.
    """ 
    if entity_id in self.variables:
      return 
    self.variables[entity_id] = EntityArrayVariable(self.name, self.shape,
        entity_id, self.e_type, features)


class InteractionScalarGroup(Group):
  """ Class representing a set of variables which are scalar and entity-related.
  """ 

  def __init__(self, name, e_type, weight_param, var_param, varH):
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
    super(InteractionScalarGroup, self).__init__(name, 1,
        e_type, weight_param, var_param, varH)

  def add_instance(self, entity_id, features): 
    """ Adds an instance variable of this group with the appropriate Variable
        subclass.

        Args:
          entity_id: the id of the entity associated to the instance.
          features: an array of features regarding the entity,
    
        Returns:
          None. The instance is included in the dictionary of instances of this
        object.
    """ 
    if entity_id in self.variables:
      return 
    self.variables[entity_id] = InteractionScalarVariable(self.name,
        entity_id, self.e_type, features)
