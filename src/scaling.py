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


""" Fits a scaler to a data.

    Args:
      scale_type: indicates the type of scale to adopt. It can be 'standard' to 
        scale with zero mean and unit standard deviation, or 'minmax' for range
        between 0 and 1.
      data: list of arrays with instances to be scaled.

    Returns:
      A scaler that fits the data.
"""
def fit_scaler(scale_type, data):
  if scale_type == 'standard':
    return StandardScaler().fit(data)
  if scale_type == 'minmax':
    return MinMaxScaler().fit(data)


""" Scales features from train and test, after fitting scaler on train.

    Args:
      scale_type: string with scale type, either 'standard' of 'minmax'.
      train: list of instances of the train.
      test: list of instances of the test.

    Returns:
      A pair with scaled train and test sets. 
"""
def scale_features(scale_type, train, test):
  scaler = fit_scaler(scale_type, train)
  train = scaler.transform(train)
  test = scaler.transform(test)
  return train, test

