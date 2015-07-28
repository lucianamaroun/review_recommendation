from math import exp, log


def sigmoid(value):
  try:
    return 1 / (1 + exp(-value))
  except Exception as e:
    print exp(-value)
    print 1 + exp(-value)
    print 1 / (1 + exp(-value))


def sigmoid_der1(value):
#  return exp(value) / (1 + exp(value)) ** 2
  try:
    log_value = value - 2 * log(1 + exp(value))
    return exp(log_value)
  except Exception as e:
    print exp(value)
    print 1 + exp(value)
    print 2 * log(1 + exp(value))
    print value

def sigmoid_der2(value):
#  return - exp(value) * (exp(value) - 1) / (1 + exp(value)) ** 3
  try:
    log_value = - value - 3 * log(1 + exp(value))
    return exp(log_value)
  except Exception as e:
    print exp(value)
    print 1 + exp(value)
    print 3 * log(1 + exp(value))
    print - value
