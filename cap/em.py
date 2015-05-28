from cap.e_step import perform_e_step
from cap.m_step import perform_m_step


_N_ITERATIONS = 50


def expectation_maximization(variables, votes):
  for _ in xrange(_N_ITERATIONS):
    perform_e_step(variables, votes)
    perform_m_step(variables)
