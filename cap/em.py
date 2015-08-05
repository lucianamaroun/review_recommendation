from cap.gibbs import gibbs_sample


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
  for i in xrange(20):
    print "EM iteration %d" % i
    print "E-step"
    perform_e_step(variables, votes, 100)
    print "M-step"
    perform_m_step(variables, votes)


def perform_e_step(groups, votes, n_samples):
  reset_variables_samples(groups)
  print "Gibbs Sampling"
  gibbs_sample(groups, votes, n_samples)
  print "Calculation of Empiric Stats"
  calculate_empiric_mean_and_variance(groups)


def reset_variables_samples(groups):
  for group in groups.itervalues():
    for variable in group.iter_variables():
      variable.reset_samples()


def calculate_empiric_mean_and_variance(groups):
  """ Calculates empiric mean and variance of the groups from samples.

      Args:
        groups: dictionary of VariableGroup objects.

      Returns:
        None. The values of mean and variance are updated on each Variable
      object.
  """
  for group in groups.itervalues():
    for variable in group.iter_variables():
      variable.calculate_empiric_mean()
      variable.calculate_empiric_var()


def perform_m_step(groups, votes):
  optimize_parameters(groups, votes)
  
def optimize_parameters(groups, votes):
  for group in groups.itervalues():
    group.weight_param.optimize(group)
    group.var_param.optimize(group)
  groups.itervalues().next().var_H.optimize(groups, votes)
