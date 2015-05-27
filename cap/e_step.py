from cap.gibbs import gibbs_sample


_N_SAMPLES = 30


def perform_e_step(variables, votes):
  gibbs_sample(variables, votes, _N_SAMPLES)
  calculate_empiric_mean_and_variance(variables)

def calculate_empiric_mean_and_variance(variables):
  """ Calculates empiric mean and variance of the variables from samples.

      Args:
        variables: dictionary of VariableGroup objects.

      Returns:
        None. The values of mean and variance are updated on each Variable
      object.
  """
  for variable_group in variables.itervalues():
    for variable in variable_group.iter_instances():
      variable.calculate_empiric_mean()
      variable.calculate_empiric_var()
