import numpy as np
from scipy.optimize import fsolve

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
def newton_raphson(fun, der, variable_group, theta_0, n_iter=50, eps=1e-8):
  return fsolve(fun, theta_0, variable_group, der)
#  for i in range(theta_0.shape[0]):
#    print "%f " % theta_0[i],
#  print ""
#  theta_n = theta_0 - np.linalg.pinv(der(theta_0, variable_group)) \
#      .dot(fun(theta_0, variable_group))
#  i = 1
#  while i < n_iter and abs(sum(theta_0 - theta_n)) > eps:
#    print "Newton-Raphson iteration %d" % i
#    print "theta_0: ",
#    for j in range(theta_0.shape[0]):
#      print "%f " % theta_0[j],
#    print ""
#    theta_0 = theta_n
#    theta_n = theta_0 - np.linalg.pinv(der(theta_0, variable_group)) \
#        .dot(fun(theta_0, variable_group))
#    i += 1
#  return theta_n 

