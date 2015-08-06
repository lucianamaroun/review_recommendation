""" Auxiliary Module
    ----------------

    Contains functions and methods which may be used in other modules.

    Functions:
      - Sigmoid
      - First derivative of sigmoid
      - Second derivative of sigmoid
      - Newton-Raphson method for Root Finding
"""

from math import exp, log
from sys import float_info

def sigmoid(value):
  try:
    denom = 1 + exp(-value)
  except OverflowError:
    return 0
  except Exception:
    return 1
  try:
    return 1 / denom
  except OverflowError:
    return float_info.max
  except Exception:
    return 0

def sigmoid_der1(value):
  # res = exp(-value) / (1 + exp(-value)) ** 2 
  try:
    e_val = exp(-value)
  except OverflowError:
    return 0
  except Exception:
    return 0
  try:
    denom = (1 + e_val) ** 2 
  except OverflowError:
    return 0
  try:
    return e_val / denom
  except OverflowError:
    return float_info.max
  except Exception:
    return 0

def sigmoid_der2(value):
  # res = exp(-value) * (exp(-value) - 1) / (1 + exp(-value)) ** 3
  try:
    e_val = exp(-value)
    e_val_inc = e_val + 1
    e_val_dec = e_val - 1
  except OverflowError:
    return 0
  except Exception:
    return 0
  try:
    res = e_val / e_val_inc
    res *= e_val_dec / e_val_inc
    return res * 1 / e_val_inc
  except OverflowError:
    return float_info.max
  except Exception:
    return 0

def newton_raphson(fun, der, variable_group, theta_0, n_iter=50, eps=1e-8):
  """ Applies Newton-Raphson's Method. This method finds an approximation for a 
      root of a function numerically by continuously updating acording to the
      derivative and the function at the current value of the variable. 
      
      Obs: In our case, it finds the optimal value of a function by finding the
      root of its derivative, thus it uses the first and the second order
      derivative.

      Args:
        fun: function which evaluates over theta to calculate the root of.
        der: function which evaluates over theta and represents the derivative.
        theta_0: value of initial theta.
        n_iter: number of iterations to perform.
  """
#  return fsolve(fun, theta_0, variable_group, der)
  theta_n = theta_0 - np.linalg.pinv(der(theta_0, variable_group)) \
      .dot(fun(theta_0, variable_group))
  i = 1
  while i < n_iter and abs(sum(theta_0 - theta_n)) > eps:
    theta_0 = theta_n
    theta_n = theta_0 - np.linalg.pinv(der(theta_0, variable_group)) \
        .dot(fun(theta_0, variable_group))
    i += 1
  return theta_n 

