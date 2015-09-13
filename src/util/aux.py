from numpy import inner
from numpy.linalg import norm
from math import sqrt

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
    return float('nan')
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
 

def vectorize(self, dict_a, dict_b):
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
