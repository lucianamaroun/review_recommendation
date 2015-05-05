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
from src.user_modeling import get_similar_users
from src.parser import parse_trusts
from cap.models import ParameterCollection, VariableCollection
from cap import constants
from cap.em import expectation_maximization

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
    variables.add_beta_variable(r_id, map_review_features(reviews[r_id]))
    variables.add_v_variable(r_id, map_review_features(reviews[r_id]))
    variables.add_xi_variable(a_id, map_author_features(users[a_id]))
    variables.add_alpha_variable(v_id, map_voter_features(users[v_id]))
    variables.add_u_variable(v_id, map_voter_features(users[v_id]))
  for author_voter, features in users_sim.items():
    variables.add_gamma_variable(author_voter, map_users_sim_features(features))
  for author_voter, features in users_conn.items():
    variables.add_lambda_variable(author_voter,
        map_users_conn_features(features))
  return variables


def map_review_features(review):
  new_review = np.array([review['num_tokens'], review['num_sents'], 
      review['uni_ratio'], review['avg_sent'], review['cap_sent'],
      review['noun_ratio'], review['adj_ratio'], review['comp_ratio'],
      review['verb_ratio'], review['adv_ratio'], review['fw_ratio'],
      review['sym_ratio'], review['noun_ratio'], review['punct_ratio'],
      review['kl'], review['pos_ratio'], review['neg_ratio']])
  new_review = np.reshape(new_review, (17, 1))
  return new_review


def map_author_features(author):
  new_author = np.array([author['num_reviews'], author['avg_rating'],
      author['num_trustors'], author['num_trustees'], author['pagerank']])
  new_author = np.reshape(new_author, (5, 1))
  return new_author


def map_voter_features(voter):
  new_voter = np.array([voter['num_trustors'], voter['num_trustees'],
      voter['pagerank'], voter['avg_rating'], voter['avg_rating_dir_net'],
      voter['avg_rating_sim'], voter['avg_help_giv'],
      voter['avg_help_giv_tru_net'], voter['avg_help_giv_sim']])
  new_voter = np.reshape(new_voter, (9, 1))
  return new_voter


def map_users_sim_features(users_sim):
  new_users_sim = np.array([users_sim['common_rated'], users_sim['jacc_rated'],
      users_sim['cos_ratings'], users_sim['pear_ratings'],
      users_sim['diff_avg_ratings'], users_sim['diff_max_ratings'],
      users_sim['diff_min_ratings']])
  new_users_sim = np.reshape(new_users_sim, (7, 1))
  return new_users_sim


def map_users_conn_features(users_conn):
  new_users_conn = np.array([users_conn['jacc_trustees'],
      users_conn['jacc_trustors'], users_conn['adamic_adar_trustees'],
      users_conn['adamic_adar_trustors']#, users_conn['katz']
      ]) 
  new_users_conn = np.reshape(new_users_conn, (4, 1))
  return new_users_conn


def main():
  import pickle
  print 'Reading pickles'
 # reviews, users, _, train, _ = model()
 # pickle.dump(reviews, open('pkl/reviews.pkl', 'w'))
 # pickle.dump(users, open('pkl/users.pkl', 'w'))
 # pickle.dump(train, open('pkl/train.pkl', 'w'))
  reviews = pickle.load(open('pkl/reviews.pkl', 'r'))
  users = pickle.load(open('pkl/users.pkl', 'r'))
  train = pickle.load(open('pkl/train.pkl', 'r'))
  
  similar = get_similar_users(users)
 # trusts = parse_trusts()
  pickle.dump(similar, open('pkl/similar.pkl', 'w'))
 # pickle.dump(trusts, open('pkl/trusts.pkl', 'w'))
 # similar = pickle.load(open('pkl/similar.pkl', 'r'))
  trusts = pickle.load(open('pkl/trusts.pkl', 'r'))
  
  print 'Modeling interaction'
  sim_author_voter = model_author_voter_similarity(train, users, similar)
  pickle.dump(sim_author_voter, open('pkl/sim_author_voter.pkl', 'w'))
  #sim_author_voter = pickle.load(open('pkl/sim_author_voter.pkl', 'r'))
  #conn_author_voter = model_author_voter_connection(train, users, trusts)
  #pickle.dump(conn_author_voter, open('pkl/conn_author_voter.pkl', 'w'))
  conn_author_voter = pickle.load(open('pkl/conn_author_voter.pkl', 'r'))
  
  print 'Creating variables'
  variables = create_variables(reviews, users, train, sim_author_voter,
      conn_author_voter)
  parameters = ParameterCollection()
  
  print 'Running EM'
  expectation_maximization(variables, parameters, train)


if __name__ == '__main__':
  main() 
