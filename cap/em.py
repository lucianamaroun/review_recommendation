from cap.e_step import perform_e_step
from cap.m_step import perform_m_step


def expectation_maximization(variables, votes):
  for i in xrange(5):
    print "EM iteration %d" % i
    print "E-step"
    perform_e_step(variables, votes, 5)
    print "M-step"
    perform_m_step(variables, votes)
  for i in xrange(5):
    print "EM iteration %d" % i
    print "E-step"
    perform_e_step(variables, votes, 20)
    print "M-step"
    perform_m_step(variables, votes)
#  for i in xrange(20):
#    print "EM iteration %d" % i
#    print "E-step"
#    perform_e_step(variables, votes, 100)
#    print "M-step"
#    perform_m_step(variables, votes)
