def perform_m_step(groups, votes):
  optimize_parameters(groups, votes)
  
def optimize_parameters(groups, votes):
  for group in groups.itervalues():
    group.weight_param.optimize(group)
    group.var_param.optimize(group)
  groups.itervalues().next().var_H.optimize(groups, votes)
