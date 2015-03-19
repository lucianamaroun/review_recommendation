from numpy.random import normal, multivariate_normal

""" Performs Gibbs Sampling over variables.

    Observation: each Variable object has a value and a list of samples. Once a
    new sample is generated, the value is updated to this new sample and the
    subsequent calculations of mean and variance of other values use this new
    value.

    Args:
      variables: a VariableCollection object.
      parameters: a ParameterCollection object.
      n_samples: the number of samples to obtain.

    Returns:
      None. The samples are inserted into Variable objects.
"""
def gibbs_sample(variables, parameters, votes, n_samples):
  
  for _ in xrange(n_samples):
    for review_id, beta in variables.beta: 
      r_votes = [vote for vote in votes if vote['review'] == review_id]
      mean, var = variables.get_beta_mean_and_variance(review_id, r_votes, 
          parameters)
      beta.value = normal(mean, sqrt(var))
      beta.samples.append(beta.value)
    
    for voter_id, alpha in variables.beta: 
      v_votes = [vote for vote in votes if vote['voter'] == voter_id]
      mean, var = variables.get_alpha_mean_and_variance(voter_id, v_votes,
          parameters)
      alpha.value = normal(mean, sqrt(var))
      alpha.samples.append(alpha.value)
    
    for author_id, xi in variables.beta: 
      a_votes = [vote for vote in votes if vote['reviewer'] == author_id]
      mean, var = variables.get_xi_mean_and_variance(author_id, a_votes,
          parameters)
      xi.value = normal(mean, sqrt(var))
      xi.samples.append(xi.value)
   
    for author_voter, gamma in variables.gamma:
      author_id, voter_id = author_voter
      av_votes = [vote for vote in votes if vote['reviewer'] == author_id and
          vote['voter'] == voter_id]
      mean, var = variables.get_gamma_mean_and_variance(author_voter, av_votes,
          parameters)
      gamma.value = normal(mean, sqrt(var))
      gamma.samples.append(gamma.value)
   
    for author_voter, lambd in variables.lambd:
      author_id, voter_id = author_voter
      av_votes = [vote for vote in votes if vote['reviewer'] == author_id and
          vote['voter'] == voter_id]
      mean, var = variables.get_lambda_mean_and_variance(author_voter, av_votes,
          parameters)
      lambd.value = normal(mean, sqrt(var))
      lambd.samples.append(lambd.value)
   
    for voter_id, u in variables.u:
      v_votes = [vote for vote in votes if vote['voter'] == voter_id]
      mean, var = variables.get_u_mean_and_variance(author_voter, v_votes,
          parameters)
      u.value = multivariate_normal(mean, var)
      u.samples.append(u.value)

    for review_id, v in variables.v:
      r_votes = [vote for vote in votes if vote['voter'] == voter_id]
      mean, var = variables.get_v_mean_and_variance(author_voter, r_votes,
          parameters)
      v.value = multivariate_normal(mean, var)
      v.samples.append(v.value)
