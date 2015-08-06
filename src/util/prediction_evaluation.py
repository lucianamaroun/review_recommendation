""" Prediction evaluation module
    ----------------------------

    Includes metrics of prediction evaluations.

    Usage:
      Used only as a module, not directly callable.
"""


from math import sqrt

from sklearn.linear_model import LinearRegression


""" Calculates squared errors.

    Args:
      result: a list of predicted values assigned to the instances.
      truth: a list with the corrected values.

    Returns:
      A list of squared errros, one for each instance.
"""
def calculate_squared_errors(result, truth):
  errors = [(result[i] - truth[i]) ** 2 for i in range(len(result))]
  return errors


""" Calculates RMSE (Root Mean Squared Error) of a prediction task.

    Args:
      result: a list of predicted values assigned to the instances.
      truth: a list with the corrected values.

    Returns:
      A real value representing the RMSE.
"""
def calculate_rmse(result, truth):
  errors = calculate_squared_errors(result, truth)
  rmse = sqrt(float(sum(errors)) / len(errors))
  return rmse


""" Calculates the accuracy of a prediction task.

    Args:
      result: a list with the predicted values for the instances.
      truth: a list with the correct values.

    Returns:
      A real value representing the accuracy of the method.
"""
def calculate_accuracy(result, truth):
  errors = calculate_squared_errors(result, truth)
  binary_errors = [1 if e > 0 else 0 for e in errors]
  accuracy = float(sum(binary_errors)) / len(binary_errors)
  return 1 - accuracy


""" Calculates R-squared of linear regression. 

    Args:
      pred: a fitted linear regression predictor.
      data: a list of instances represented as a set of features.
      truth: a list of truth values associated to instances.

    Returns:
      A real value containing the R-squared.
"""
def calculate_rsquared(pred, data, truth):
  return pred.score(data_np, truth)

