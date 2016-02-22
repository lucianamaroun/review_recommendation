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
    if len(pred) == 0:
      continue
    best = 0
    for j in xrange(1, len(pred)):
      if pred[j] > pred[best]:
        best = j
    top.append(truth[best])
    pred = pred[:best] + pred[best+1:]
    truth = truth[:best] + truth[best+1:]
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
      # zero-relevance on top is bad (conservative)

def calculate_avg_ndcg(votes, reviews, pred, truth, k):
  """ Calculates the average nDCG by grouping votes in (user, product) pairs in
      order to compose rankings.

      Args:
        votes: set of votes to evaluate through grouping and ranking.
        reviews: dictionary of reviews.
        pred: list of floats with predicted relevances for each vote in votes.
        truth: list of floats (or integers) with true relevances for each vote
      in votes.
        k: an integer with the limit position of the ranking to calculate the
      score.

      Returns:
        A float in range [0, 1] with nDCG@K score.
  """
  pred_group = {}
  truth_group = {}
  for i, vote in enumerate(votes):
    voter = vote['voter']
    product = reviews[vote['review']]['product']
    key = (voter, product)
    if key not in pred_group:
      pred_group[key] = []
      truth_group[key] =[]
    pred_group[key].append(pred[i])
    truth_group[key].append(truth[i])
  score_sum = 0.0
  for key in pred_group:
    ndcg = calculate_ndcg(pred_group[key], truth_group[key], k)
    score_sum += ndcg
  score = score_sum / len(pred_group)
  return score


def calculate_ap(pred, truth, k):
  """ Gets the average precision.

      Observation:
      - We consider the top-k in true ranking as relevant items. Consquently,
      there are always k relevant items.

      Args:
        pred: list of floats with predicted relevances.
        truth: list of floats (or integers) with true relevances.
        k: integer with the size of ranking to consider.

      Returns:
        A real value with the average precition at k.      
  """
  i_ranking = sorted(range(len(truth)), key=lambda p: pred[p], reverse=True)
 # p_ranking = sorted(range(len(pred)), key=lambda p: pred[p], reverse=True)
 # top = t_ranking[:k]
 # worst_rel = truth[top[-1]]
 # for pos in xrange(k, len(t_ranking)):
 #   index = t_ranking[pos]
 #   if truth[index] == worst_rel:
 #     top.append(index)
 #   else:
 #     break
  avg_prec = 0.0 
  acc = 0.0
  for i in xrange(k):
   # if p_ranking[i] in top:
    if truth[i_ranking[i]] == 5:
      acc += 1.0
      avg_prec += (acc / (i+1))
  avg_prec /= k
  return avg_prec 


def calculate_map(votes, reviews, pred, truth, k):
  """ Calculates the mean average precision (MAP) by grouping votes in 
      (user, product) pairs in order to compose rankings.

      Args:
        votes: set of votes to evaluate through grouping and ranking.
        reviews: dictionary of reviews.
        pred: list of floats with predicted relevances for each vote in votes.
        truth: list of floats (or integers) with true relevances for each vote
      in votes.
        k: an integer with the limit position of the ranking to calculate the
      score.

      Returns:
        A float in range [0, 1] with MAP@K score.
  """
  pred_group = {}
  truth_group = {}
  for i, vote in enumerate(votes):
    voter = vote['voter']
    product = reviews[vote['review']]['product']
    key = (voter, product)
    if key not in pred_group:
      pred_group[key] = []
      truth_group[key] =[]
    pred_group[key].append(pred[i])
    truth_group[key].append(truth[i])
  score_sum = 0.0
  for key in pred_group:
    ap = calculate_ap(pred_group[key], truth_group[key], k)
    score_sum += ap 
  score = score_sum / len(pred_group)
  return score

def probability_stop(relevance):
  """ Probability of user stop search given an item of a certain relevance.

      Args:
        relevance: relevance grade of the item.

      Returns:
        A value in [0, 1] with probability mapping of relevance.
  """
  return (2.0**relevance - 1.0) / 32.0 # denominator 2**5 

def calculate_err(pred, truth):
  """ Gets the expected reciprocal rank.

      Args:
        pred: list of floats with predicted relevances.
        truth: list of floats (or integers) with true relevances.

      Returns:
        A real value with the ERR.
  """
  p_ranking = sorted(range(len(pred)), key=lambda p: pred[p], reverse=True)
  err = 0.0 
  p = 1.0
  for i in xrange(len(p_ranking)):
    relevance = truth[p_ranking[i]]
    local_p = probability_stop(relevance)
    err += p * local_p / (i + 1.0)
    p *= (1.0 - local_p)
  return err 
