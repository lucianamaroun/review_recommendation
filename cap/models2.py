from numpy import array, random, reshape

from cap import constants as const
from cap.aux import sigmoid, sigmoid_der1, sigmoid_der2
from cap.newton_raphson import newton_raphson


class Value(object):
  """ Class representing a value in the algorithm, either a scalar or a matrix.
  """

  def __init__(self, shape):
    """ Constructor of value. Any value has a shape and a value, which is 
        randomly initialized.

        Args:
          shape: a positive integer or a 2-tuple of positive integers specifying
            the shape of a scalar or a matrix, respectively.
    """
    self.shape = shape
    self._initialize_value()

  def _initialize_value(self):
    """ Initilizes the value according to the shape. The value can be a scalar, 
        in which case the shape is not a tuple, or a matrix. The added values
        are random.
        
        Obs:
        - To assure the values are bigger than 1, a small positive value is
        added.
        - If the shape is not a single positive integer nor a tuple of
        positive integers, an exception is raised and the execution is
        interrupted.
    """
    try:
      if type(self.shape) is int and self.shape > 0:
        self.value = random.random() + 0.000001
      elif type(self.shape[0]) is int and type(self.shape[1]) is int and \
          self.shape[0] > 0 and self.shape[1] > 0:
        self.value = array([[random.random() + 0.0000001 for _ in
            range(self.shape[1])] for _ in range(self.shape[0])])
      else:
        raise TypeError()
    except TypeError as e:
      print e
      print 'TypeError: shape should be a positive int or a 2-tuple of positive ints.'
      import sys
      sys.exit()

  def update_value(self, value):
    """ Updates the value.
       
        Args:
          value: value to replace the old one.
        
        Obs: 
        - If the shape is the new value is not the same as the previous one, an
        exception is raised and the execution is interrupted.
    """
    if (type(value) != type(self.value)):
      print 'TypeError: value should have type %s' % type(self.value)
      import sys
      sys.exit()
    elif (hasattr(value, 'shape') and self.shape != value.shape):
      print 'TypeError: value should have shape (%d, %d)' % self.shape
      import sys
      sys.exit()
    self.value = value 


class Parameter(Value):
  """ Class specifying a Parameter, which defines latent variables distributions
      and relationship with observed variables.  
  """

  def __init__(self, name, shape):
    """ Constructor of Parameter. The values are randomly initialized using
        Uniform(0, 1).

        Args:
          name: string with the name of the parameter.
          size: pair with vector or matrix dimensions.

        Returns:
          None.
    """
    self.name = name
    super(Parameter, self).__init__(shape)


class Variable(Value):
  """ Class defining a latent variable.
  """
  
  def __init__(self, shape, entity_id, features): 
    """ Constructor of Variable.
    
        Args:
          shape: a tuple of positive integers or a positive integer with the
            shape of the variable.
          entity_id: id of the entity holding this variable.
          features: set of observed features associated with the variable.
          
        Returns:
          None.
    """
    super(Variable, self).__init__(shape)
    self.entity_id = entity_id
    self.features = reshape(features, (features.shape[0], 1))
    self.samples = []
    self.cond_mean = None
    self.cond_var = None
    self.empiric_mean = None
    self.empiric_var = None


class VariableGroup(object):
  """ Class implementing a set of variables of the same type but regarding 
      different entities.
  """
  
  def __init__(self, name, shape, weight_param, var_param):
    """ Initializes the group of variables.

        Args:
          name: a string with the name of the group.
          shape: the shape of each variable instance.
          weight_param: parameter object of regression weights.
          var_param: variance parameter of this variable's distribution.
    """
    self.name = name
    self.weight_param = weight_param
    self.var_param = var_param
    self.variables = []
    
  def add_instance(self, entity_id, features)  
    self.variables.append(Variable(self.shape, entity_id, features))
