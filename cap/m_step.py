def perform_m_step(groups):
  optimize_parameters(groups)
  
def optimize_parameters(groups):
  for group in groups.itervalues():
    group.weight_param.optimize(group.iter_variables())
    group.var_param.optimize(group.iter_variables())
