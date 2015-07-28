import numpy as np

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
def newton_raphson(fun, der, variable_group, theta_0, n_iter=5):
  for _ in xrange(n_iter):
    print theta_0
    print der
    print der(theta_0, variable_group)
    print fun
    print fun(theta_0, variable_group)
    theta_n = theta_0 - np.linalg.pinv(der(theta_0, variable_group)) \
        .dot(fun(theta_0, variable_group))
    theta_0 = theta_n
    print theta_0
  return theta_n

