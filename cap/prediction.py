""" Prediction Module for CAP Baseline
    ----------------------------------

    This module implements a Monte Carlo EM algorithm for fitting latent
    variables and data distributions' parameters.

    Usage:
      $ python -m cap.prediction
    on root directory of the project.
"""

import numpy as np

from src.modeling import model
from src.author_voter_modeling import model_author_voter_similarity, \
    model_author_voter_connection

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
def create_variables(reviews, users, votes, users_sim, users_conn):
  variables = VariableCollection()
  for vote in votes:
    r_id, a_id, v_id = vote['review'], vote['reviewer'], vote['voter']
    variables.add_beta_variable(r_id, map_review_features(review[r_id]))
    variables.add_v_variable(r_id, map_review_features(review[r_id]))
    variables.add_xi_variable(a_id, map_author_features(users[a_id]))
    variables.add_alpha_variable(v_id, map_voter_features(users[v_id]))
    variables.add_u_variable(v_id, map_voter_features(users[v_id]))
  for author_voter, features in author_voter_sim.items():
    variables.add_gamma_variable(author_voter, features)
  for author_voter, features in author_voter_conn.items():
    variables.add_lambda_variable(author_voter, features)
  return variables


def map_review_features(review):
  new_review = np.array([review['num_token'], review['num_sent'], 
      review['unique_ratio'], review['avg_sent'], review['cap_ratio'],
      review['noun_ratio'], review['adj_ratio'], review['comp_ratio'],
      review['verb_ratio'], review['adv_ratio'], review['fw_ratio'],
      review['sym_ratio'], review['noun_ratio'], review['punct_ratio'],
      review['kl_div'], review['pos_ratio'], review['neg_ratio']])
  np.reshape(new_review, (17, 1))
  return new_review


def map_author_features(author):
  new_author = np.array([author['num_reviews'], author['avg_rating'],
      author['num_trustors'], author['num_trustees'], author['pagerank']])
  np.reshape(new_author, (5, 1))
  return new_author


def map_voter_features(voter):
  new_author = np.array([voter['num_trustors'], voter['num_trustees'],
      voter['pagerank'], voter['avg_rating'], voter['avg_rating_dir_net'],
      voter['avg_rating_sim'], voter['avg_help_giv'],
      voter['avg_help_giv_tru_net'], voter['avg_help_giv_sim']])
  np.reshape(new_author, (9, 1))
  return new_author


def main():
  reviews, users, _, train, _ = model()
  sim_author_voter = model_author_voter_similarity(train, users)
  conn_author_voter = model_author_voter_connection(train, users)
  variables = create_variables(reviews, users, train, users_sim, users_conn)
  
