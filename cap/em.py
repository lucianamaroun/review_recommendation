_N_SAMPLES = 30

def e_step(variables, parameters, votes, sim, trust):
  gibbs_sampling(variables, parameters, voters, sim, trust, n_samples)
  variables.calculate_empiric_mean_and_variance()

def m_step(variables, parameters, votes):
  pass
