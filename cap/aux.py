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

