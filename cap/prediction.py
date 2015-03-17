""" Prediction Module for CAP Baseline
    ----------------------------------

    This module implements a Monte Carlo EM algorithm for fitting latent
    variables and data distributions' parameters.

    Usage:
      $ python -m cap.prediction
    on root directory of the project.
"""

from cap.models import ParameterCollection, VariableCollection
from cap import constants


""" Creates latent variables associated to each entity.

    Args:
      reviews: dictionary of reviews' features indexed by id.
      authors: dictionary of authors' features indexed by id.
      reviews: dictionary of voters' features indexed by id.
    
    Returns:
      A VariableCollection object,
"""
def create_variables(reviews, authors, voters, author_voter_pairs):
  variables = VariableCollection()
  for review_id, features in reviews.items():
    variables.add_beta_variable(review_id, features)
    variables.add_v_variable(review_id, features)
  for author_id, features in authors.items():
    variables.add_xi_variable(author_id, features)
  for voter_id, features in voters.items():
    variables.add_alpha_variable(voter_id, features)
    variables.add_u_variable(voter_id, features)
  for author_voter, features in author_voter_pairs.items():
    variables.add_gamma_variable(author_voter, features)
    variables.add_lambda_variable(author_voter, features)
  return variables


def main():
  reviews, authors, voters, author_voter_pairs = obtain_features()
  variables = create_variables(reviews, authors, voters, author_voter_pairs)
  
