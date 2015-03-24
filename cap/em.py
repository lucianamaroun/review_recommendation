from cap.gibbs import gibbs_sample

_N_SAMPLES = 30

def e_step(variables, parameters, votes):
  gibbs_sample(variables, parameters, votes, _N_SAMPLES)
  variables.calculate_empiric_mean_and_variance()

def m_step(variables, parameters, votes):
  pass

def expectation_maximization(variables, parameters, votes):
  e_step(variables, parameters, votes)
  m_step(variables, parameters, votes)
