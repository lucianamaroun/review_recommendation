from cap.gibbs import gibbs_sample

_N_SAMPLES = 30
_N_ITERATIONS = 50

def e_step(variables, parameters, votes):
  gibbs_sample(variables, parameters, votes, _N_SAMPLES)
  variables.calculate_empiric_mean_and_variance()

def m_step(variables, parameters):
  parameters.adjust(variables) 

def expectation_maximization(variables, parameters, votes):
  for _ in xrange(_N_ITERATIONS):
    e_step(variables, parameters, votes)
    m_step(variables, parameters)
