""" Scaling Module
    --------------

    Fits a scaler to training data and applies to train and test. There are two
    options of scalers: (1) a standard scaler, obtaining zero mean and unit
    variance for features; and (2) a minmax scaler, which maps features ranges
    to interval [0, 1].

    Usage:
      Used only as a module, not directly callable.
"""


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from numpy import array, hstack, hsplit


def group_by_qid(data, qid):
  """ Groups instances by query id (user).
      
      Args:
        data: 2-D array with instances in lines and features in columns.
        qid: list of query ids associated to each instance, in the same order.

      Returns:
        A dictionary indexed by query id and containing a list of instances
      arrays associated with it.
  """
  grouped = {}
  for i in xrange(data.shape[0]):
    if qid[i] not in grouped:
      grouped[qid[i]] = []
    grouped[qid[i]].append(data[i,:])
  return grouped


def fit_scaler(scale_type, data):
  """ Fits a scaler to a data.

      Args:
        scale_type: indicates the type of scale to adopt. It can be 'standard' to 
          scale with zero mean and unit standard deviation, or 'minmax' for range
          between 0 and 1.
        data: list of arrays with instances to be scaled.

      Returns:
        A scaler that fits the data.
  """
  if scale_type == 'standard':
    return StandardScaler(copy=False).fit(data)
  if scale_type == 'minmax':
    return MinMaxScaler(copy=False).fit(data)


def fit_scaler_by_query(scale_type, data, qid, qid_dep_size):
  """ Fits a scaler to data.

      Args:
        scale_type: indicates the type of scale to adopt. It can be 'standard' to 
          scale with zero mean and unit standard deviation, or 'minmax' for range
          between 0 and 1.
        data: list of arrays with instances to be scaled.
        qid: list of query ids associated to each instance.
        qid_dep_size: number of features query-dependent, which are in the end
          of the instance array of features.

      Returns:
        A scaler that fits the data.
  """
  data = array(data)
  dim = data.shape[1]
  q_undep, q_dep = hsplit(data, [dim-qid_dep_size])
  if scale_type == 'standard':
    overall_scaler = StandardScaler(copy=False).fit(q_undep)
  if scale_type == 'minmax':
    overall_scaler = MinMaxScaler(copy=False).fit(q_undep)
  return overall_scaler


def scale_features(scaler, data, qid=None, qid_dep_size=None, 
    scale_type='minmax'):
  """ Scales features from data given a fitted scaler. 

      Args:
        scaler: a scaler object with transform function.
        data: list of instances to scale. 
        qid: list of query ids associated to each instance.
        qid_dep_size: number of features query-dependent, which are in the end
          of the instance array of features.
        scale_type: indicates the type of scale to adopt. It can be 'standard' to 
          scale with zero mean and unit standard deviation, or 'minmax' for range
          between 0 and 1.

      Returns:
        A pair with scaled train and test sets. 
  """
  data = array(data)
  if qid is None:
    data = scaler.transform(data)
  else:
    dim = data.shape[1]
    overall_scaler = scaler 
    q_undep, q_dep = hsplit(data, [dim-qid_dep_size])
    q_undep = overall_scaler.transform(q_undep)
    qid_grouped = group_by_qid(q_dep, qid)
    q_scalers = {}
    for q in qid_grouped: 
      if scale_type == 'minmax':
        q_scalers[q] = MinMaxScaler(copy=False).fit(qid_grouped[q])
      if scale_type == 'standard':
        q_scalers[q] = StandardScaler(copy=False).fit(qid_grouped[q])
    for i in xrange(q_dep.shape[0]):
      q_dep[i] = q_scalers[qid[i]].transform([q_dep[i]])
    data = hstack((q_undep, q_dep))
  return data 

