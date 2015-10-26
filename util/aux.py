""" Auxiliary Module
    ----------------

    Auxiliary model used across modules. Contain metrics and computations which
    may be used by any method.

    Not directly callable.
"""


from math import exp, sqrt, log
from sys import float_info

from numpy import zeros, nan, isnan
from numpy.linalg import norm
from scipy.special import expit


_LAMBDA = 100


def cosine(vector_a, vector_b):
  """ Calculates the cosine similarity between two vectors in the same
      n-dimensional space.

      Args:
        vector_a: a numpy array with one vector.
        vector_b: a numpy array with the other vector.

      Returns:
        A float with the cosine similarity value, in range [-1, 1].
  """
  if (vector_a.size != vector_b.size):
    return nan 
  if (vector_a.size == 0):
    return 0.0
  v_a = vector_a.reshape(vector_a.size)
  v_b = vector_b.reshape(vector_b.size)
  cos = v_a.dot(v_b)
  norms = (sqrt(v_a.dot(v_a)) * sqrt(v_b.dot(v_b)))
  if norms == 0:
    return 0.0
  cos = cos / norms
  return cos
 

def vectorize(dict_a, dict_b):
  """ Maps two dictionary of values into two vectors in a common space. Each
      unique key defines a dimension; if a key is absent, the value is interpreted
      as zero.

      Args:
        dict_a: dictionary containing the one set of values.
        dict_b: dictionary containing the another set of values.

      Returns:
        Two numpy arrays with the values in the common space. The number of
      dimensions is defined by the size of the union of dictionary keys.
  """
  dimensions = set(dict_a.keys()).union(set(dict_b.keys()))
  vec_a = zeros(len(dimensions))
  vec_b = zeros(len(dimensions))
  for dim_index, dim_name in enumerate(dimensions):
    vec_a[dim_index] = dict_a[dim_name] if dim_name in dict_a else 0
    vec_b[dim_index] = dict_b[dim_name] if dim_name in dict_b else 0
  return vec_a, vec_b


def shrunk_cosine(votes_a, votes_b):
  """ Calculates cosine similarity between two dictionary of votes. First,
      ratings are projected to vectors in the space of intersection of voted
      objects. Then, cosine is calculated with shrunking, where the lenght of
      each vector is the length of evaluation intersection.
      
      Args:
        votes_a: dictionary of votes of entity a, indexed by voted object id.
        votes_b: dictionary of votes of entity b, indexed by voted object id.

      Returns:
        A float with cosine similarity.
  """
  a_vec, b_vec = vectorize(votes_a, votes_b)
  cos = cosine(a_vec, b_vec)
  intersection = len(set(votes_a.keys()).intersection(set(voter_b.keys())))
  if isnan(cos):
    return 0.0
  return intersection / (intersection + _LAMBDA) * cosine(a_vec, b_vec)


def shrunk_pearson(votes_a, votes_b):
  """ Calculates Pearson similarity between two dictionary of votes. First,
      ratings are projected to vectors in the space of intersection of voted
      objects. Then, Pearson is calculated with shrunking, where the lenght of
      each vector is the length of evalation intersection.
      
      Args:
        votes_a: dictionary of votes of entity a, indexed by voted object id.
        votes_b: dictionary of votes of entity b, indexed by voted object id.

      Returns:
        A float with cosine similarity.
  """
  a_vec, b_vec = vectorize(votes_a, votes_b)
  pear = pearsonr(a_vec, b_vec)[0]
  intersection = len(set(votes_a.keys()).intersection(set(voter_b.keys())))
  if isnan(pear):
    return 0.0
  return intersection / (intersection + _LAMBDA) * pear


def sigmoid(value):
  """ Computes the sigmoid function applied to value. The function corresponds
      to 1 / (1 + exp(-value))

      Observation:
      - If an overflow or underflow is obtained ate each step of the
        computation, then a asymptotic approximate value is returned.

      Args:
        value: a float

      Returns:
        A float with the function return value.
  """
  return expit(value)
#  try:
#    val_inc = 1.0 + exp(-value)
#  except OverflowError:
#    return 0.0
#  try:
#    val = 1.0 / val_inc
#  except OverflowError: # always positive, so max inf
#    return float_info.max 
#  return val 
 # try:
 #   denom = 1 + exp(-value)
 # except OverflowError:
 #   return 0
 # except Exception: #Underflow
 #   return 1
 # try:
 #   return 1 / denom
 # except OverflowError:
 #   return float_info.max
 # except Exception:
 #   return 0

def sigmoid_der1(value):
  """ Computes the sigmoid function applied to value. The function corresponds
      to exp(value) / (1 + exp(value)) ** 2

      Observation:
      - If an overflow or underflow is obtained ate each step of the
        computation, then a asymptotic approximate value is returned.

      Args:
        value: a float

      Returns:
        A float with the function return value.
  """
  e_val = expit(value)
  return e_val * (1 - e_val)
#  try:
#    val = exp(value)
#    val_inc = val + 1.0
#  except OverflowError:
#    return 0.0
#  try:
#    val = value - 2 * log(val_inc)
#  except OverflowError: # only overflows to negative infinite, log is positive
#    # because val is greater than one
#    return - float_info.max
#  return exp(val)
 # try:
 #   e_val = exp(-value)
 # except OverflowError: # Denominator higher
 #   return 0
 # except Exception: # Underflow: Nominator higher
 #   return float_info.max
 # try:
 #   denom = (1 + e_val) ** 2 
 # except OverflowError:
 #   return 0
 # except Exception:
 #   return float_info.max
 # try:
 #   return e_val / denom
 # except OverflowError:
 #   return float_info.max
 # except Exception:
 #   return 0

def sigmoid_der2(value):
  """ Computes the sigmoid function applied to value. The function corresponds
      to - exp(value) * (exp(value) - 1) / (1 + exp(value)) ** 3

      Observation:
      - If an overflow or underflow is obtained ate each step of the
        computation, then a asymptotic approximate value is returned.

      Args:
        value: a float

      Returns:
        A float with the function return value.
  """
  e_val = expit(value)
  return e_val * (2 * e_val ** 2 - 3 * e_val + 1)
#  try:
#    val = exp(value)
#    val_inc = val + 1.0
#    val_dec = val - 1.0
#  except OverflowError:
#    return 0.0
#  if val_dec < 0:
#    try:
#      val = value + log(- val_dec) - 3 * log(val_inc)
#    except OverflowError: # -val_dec is between 0 and 1, thus negative log, and
#        # val_inc is greater than one, but subtracted, thus the overflow is 
#        # negative
#      return - float_info.max
#    return exp(value)
#  else:
#    try:
#      val = value + log(val_dec) - 3 * log(val_inc)
#    except OverflowError:
#      # val_dec is positive but val_inc is greater and subtracted, thus only
#      # negative overflow might occur
#      return - float_info.max
#    return - exp(value)
#  try:
#    val = val / val_inc
#    val = val / val_inc
#  except OverflowError:
#    return float_into.max
#  return - val
 # try:
 #   e_val = exp(-value)
 #   e_val_inc = e_val + 1
 #   e_val_dec = e_val - 1
 # except OverflowError: # Denominator higher
 #   return 0
 # except Exception: # Underflow: Nominator higher
 #   return float_info.max
 # try:
 #   res = e_val / e_val_inc
 #   res *= e_val_dec / e_val_inc
 #   return res / e_val_inc
 # except OverflowError:
 #   return float_info.max
 # except Exception:
 #   return 0
