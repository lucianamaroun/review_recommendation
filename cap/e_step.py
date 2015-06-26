from cap.gibbs import gibbs_sample


_N_SAMPLES = 30


def perform_e_step(groups, votes):
  gibbs_sample(groups, votes, _N_SAMPLES)
  calculate_empiric_mean_and_variance(groups)

def calculate_empiric_mean_and_variance(groups):
  """ Calculates empiric mean and variance of the groups from samples.

      Args:
        groups: dictionary of VariableGroup objects.

      Returns:
        None. The values of mean and variance are updated on each Variable
      object.
  """
  for group in groups.itervalues():
    for variable in group.iter_instances():
      variable.calculate_empiric_mean()
      variable.calculate_empiric_var()
