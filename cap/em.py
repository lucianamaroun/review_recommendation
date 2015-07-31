from cap.e_step import perform_e_step
from cap.m_step import perform_m_step


_N_ITERATIONS = 5


def expectation_maximization(variables, votes):
  for i in xrange(_N_ITERATIONS):
    print "EM iteration %d" % i
    print "E-step"
    perform_e_step(variables, votes)
    print "M-step"
    perform_m_step(variables, votes)
