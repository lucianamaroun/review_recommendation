_N_ITERATIONS = 50

def perform_m_step(variables, parameters):
  optimize_parameters(variables, parameters)
  
def optimize_parameters(variables, parameters):
  for parameter_group in parameters:
    for parameter in parameter_group:
      parameter.optimize(variables)
