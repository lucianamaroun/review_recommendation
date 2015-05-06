from cap.e_step import perform_e_step
from cap.m_step import perform_m_step


def expectation_maximization(variables, parameters, votes):
  for _ in xrange(_N_ITERATIONS):
    perform_e_step(variables, parameters, votes)
    perform_m_step(variables, parameters)
