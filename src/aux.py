from numpy import inner
from numpy.linalg import norm
from math import sqrt

def cosine(vector_a, vector_b):
  if (vector_a.size != vector_b.size):
    return float('nan')
  if (vector_a.size == 0):
    return 0.0 # TODO: user other value?
  v_a = vector_a.reshape(vector_a.size)
  v_b = vector_b.reshape(vector_b.size)
  cos = v_a.dot(v_b)
  norms = (sqrt(v_a.dot(v_a)) * sqrt(v_b.dot(v_b)))
  if norms == 0:
    return 0.0
  cos = cos / norms
  return cos
