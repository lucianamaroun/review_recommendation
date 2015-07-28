def perform_m_step(groups):
  optimize_parameters(groups)
  
def optimize_parameters(groups, votes):
  for group in groups.itervalues():
    group.weight_param.optimize(group)
    group.var_param.optimize(group)
    h_group.
  groups[0].var_H.optimize(groups, vote)
