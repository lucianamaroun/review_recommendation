""" Metrics Module
    --------------

    Includes metrics of prediction evaluations.

    Usage:
      Used only as a module, not directly callable.
"""


from math import sqrt, log


def calculate_squared_errors(result, truth):
  """ Calculates squared errors.

      Args:
        result: a list of predicted values assigned to the instances.
        truth: a list with the corrected values.

      Returns:
        A list of squared errros, one for each instance.
  """
  errors = [(result[i] - truth[i]) ** 2 for i in range(len(result))]
  return errors


def calculate_rmse(result, truth):
  """ Calculates RMSE (Root Mean Squared Error) of a prediction task.

      Args:
        result: a list of predicted values assigned to the instances.
        truth: a list with the corrected values.

      Returns:
        A real value representing the RMSE.
  """
  errors = calculate_squared_errors(result, truth)
  rmse = sqrt(float(sum(errors)) / len(errors))
  return rmse


def calculate_accuracy(result, truth):
  """ Calculates the accuracy of a prediction task.

      Args:
        result: a list with the predicted values for the instances.
        truth: a list with the correct values.

      Returns:
        A real value representing the accuracy of the method.
  """
  errors = calculate_squared_errors(result, truth)
  binary_errors = [1 if e > 0 else 0 for e in errors]
  accuracy = float(sum(binary_errors)) / len(binary_errors)
  return 1 - accuracy


def calculate_rsquared(pred, data, truth):
  """ Calculates R-squared of linear regression. 

      Args:
        pred: a fitted linear regression predictor.
        data: a list of instances represented as a set of features.
        truth: a list of truth values associated to instances.

      Returns:
        A real value containing the R-squared.
  """
  return pred.score(data_np, truth)


def get_top_k_relevance(pred, truth, k):
  """ Gets the top-k true relevance values by sorting using the predicted
      relevance values.

      Args:
        pred: list of floats with predicted relevances.
      truth: list of floats (or integers) with true relevances.
        k: integer with the size of ranking to consider.

      Returns:
        A list with the k values with true relevances sorted by predicted
      relevances.
  """
  top = []
  for i in xrange(k):
    best = 0
    for j in xrange(1, len(pred)):
      if pred[j] > pred[best]:
        best = j
    top.append(truth[best])
    pred[best] = -float('inf')
  return top


def calculate_dcg(pred, truth, k):
  """ Calculates Discounted Cumulative Gain of a ranking until certain
      position K (DCG@K).

      Args:
        pred: list of floats with predicted relevances.
        truth: list of floats with true relevances.
        k: an integer with the limit position of the ranking to calculate the
      score.

      Returns:
        A float with the DCG@K value.
  """
  ranking = get_top_k_relevance(pred, truth, k)
  dcg = sum([(2**x - 1) / log(pos+2, 2) for pos, x in enumerate(ranking)]) 
  return dcg


def calculate_ndcg(pred, truth, k):
  """ Calculates the normalized Discounted Cumulative Gain until certain
      position K (nDCG@K). This metrics normalizes the DCG by dividing to the
      DCG of the best ranking, which sorts considering the true relevances.

      Observation:
      - If the best DGC equals 0 (which may occur with a ranking with all
        relevances equal to zero), then the value of the nDCG is considered to
        be 1.0, since there is no way to be worse than the best ranking.

      Args:
        pred: list of floats with predicted relevances.
        truth: list of floats (or integers) with true relevances.
        k: an integer with the limit position of the ranking to calculate the
      score.

      Returns:
        A float in range [0, 1] with nDCG@K score.
  """
  curr_dcg = calculate_dcg(pred, truth, k)
  best_dcg = calculate_dcg(truth, truth, k)
  return curr_dcg / best_dcg if best_dcg != 0.0 else 0.0 
      # zero-relevance on top is bad, being conservative
