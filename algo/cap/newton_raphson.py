""" Netwon-Raphson Module
    ---------------------

    Computes root of a function using Newton-Raphson method. This method is used
    in the M-step to compute the parameters of interaction variables regression,
    since it is not linear due to sigmoid function. This is equivalent to
    finding the root of the derivative of the squared error of regression in
    general case.

    Not directly callable.
"""


from numpy import  allclose, zeros, errstate, any, isnan, isinf
from scipy.linalg import pinv2 as pinv

from algo.cap.const import NR_ITER, NR_TOL, NR_STEP


def newton_raphson(fun, der, variable_group, theta_0, n_iter=NR_ITER, eps=NR_TOL, 
    step=NR_STEP):
  """ Applies Newton-Raphson's Method. This method finds an approximation for a 
      root of a function by continuously updating according to the
      derivative and the function at the current value of the variable. 
      
      Observation:
      - In CAP, it is used for finding the optimal value of a function by
      calculating an approximation of the root of its derivative, thus it uses
      the first and the second order derivative.

      Args:
        fun: function which evaluates over theta to calculate the root of.
        der: function which evaluates over theta and represents the derivative.
        theta_0: value of initial theta.
        n_iter: number of iterations to perform.
        eps: tolerance for difference to zero.
        step: update rate.

      Returns:
        The approximated value for the root of the function.
  """
  der_val = der(theta_0, variable_group)
  fun_val = fun(theta_0, variable_group)
  if any(isnan(der_val)) or any(isinf(der_val)):# or any(isnan(fun_val)) or \
   # any(isinf(fun_val)):
    return theta_0
  der_inv = pinv(der_val)
#  if any(isnan(der_inv)) or any(isinf(der_inv)):
#    return theta_0
  dot = der_inv.dot(fun_val)
#  if any(isnan(dot)) or any(isinf(dot)):
#    return theta_0
  theta = theta_0 - step * dot
#  if any(isnan(theta)) or any(isinf(theta)):
#    return theta_0
  der_val = der(theta, variable_group)
  old_fun_val = float('inf')
  fun_val = fun(theta, variable_group)
  i = 1
  while i < n_iter and not allclose(fun_val, old_fun_val, atol=eps):
    if any(isnan(der_val)) or any(isinf(der_val)):# or any(isnan(fun_val)) or \
 #     any(isinf(fun_val)):
      return theta
    der_inv = pinv(der_val)
 #   if any(isnan(der_inv)) or any(isinf(der_inv)):
 #     return theta
    dot = der_inv.dot(fun_val)
 #   if any(isnan(dot)) or any(isinf(dot)):
 #     return theta
    new_theta = theta - step * dot
 #   if any(isnan(new_theta)) or any(isinf(new_theta)):
 #     return theta
    theta = new_theta
    der_val = der(theta, variable_group)
    old_fun_val = fun_val
    fun_val = fun(theta, variable_group)
    i += 1
  return theta

