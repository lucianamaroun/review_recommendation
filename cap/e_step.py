from cap.gibbs import gibbs_sample

_N_SAMPLES = 30

def perform_e_step(variables, parameters, votes):
  gibbs_sample(variables, parameters, votes, _N_SAMPLES)
  calculate_empiric_mean_and_variance(variables)

def calculate_empiric_mean_and_variance(variables):
  """ Calculates empiric mean and variance of the variables from samples.

      Args:
        None.

      Returns:
        None. The values of mean and variance are updated on Variable object.
  """
  for variable_group in variables:
    for e_id, variable in variable_group.items():
      variable.calculate_mean(variable.samples) #TODO: method
      variable.calculate_variance(variable.samples) #TODO: method
