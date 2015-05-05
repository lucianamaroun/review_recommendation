from math import exp


def sigmoid(value):
  return 1 / (1 + exp(-value))


def sigmoid_der1(value):
  return exp(value) / (1 + exp(value)) ** 2


def sigmoid_der2(value):
  return - exp(value) * (exp(value) - 1) / (1 + exp(value)) ** 3
