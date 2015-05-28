def perform_m_step(variables):
  optimize_parameters(variables)
  
def optimize_parameters(variables):
  for variable_group in variables.itervalues():
    variable_group.weight_param.optimize(variable_group.iter_instances())
    variable_group.var_param.optimize(variable_group.iter_instances())
