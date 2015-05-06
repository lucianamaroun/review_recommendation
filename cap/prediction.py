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
from cap.models2 import VariableGroup, Parameter 
from cap import constants
from cap.em import expectation_maximization
from cap.map_features import map_review_features, map_author_features, \
    map_voter_features, map_users_sim_features, map_users_conn_features


""" Creates empty latent variable groups with its corresponding parameters.

    Args:
      reviews: dictionary of reviews' features indexed by id.
      authors: dictionary of authors' features indexed by id.
      reviews: dictionary of voters' features indexed by id.
    
    Returns:
      A dictionary of VariableGroup objects indexed by names.
"""
def create_variables():
  variables = {
    'alpha': VariableGroup('alpha', 1, Parameter('d', (9, 1)), 
      Parameter('var_alpha', 1)),
    'beta': VariableGroup('beta', 1, Parameter('g', (17, 1)), 
      Parameter('var_beta', 1)),
    'xi': VariableGroup('beta', 1, Parameter('b', (5, 1)), 
      Parameter('var_beta', 1)),
    'u': VariableGroup('u', (const.K, 1), Parameter('W', (const.K, 9)),
    Parameter('var_u', 1)),
    'v': VariableGroup('v', (const.K, 1), Paramete('V', (const.K, 17)),
      Parameter('var_v', 1)),
    'gamma': variablegroup('gamma', 1, parameter('r', (7, 1)),
      parameter('var_gamma', 1),
    'lambda': variablegroup('lambda', 1, parameter('h', (4, 1)),
      parameter('var_lambda', 1)
  }
  return variables


def populate_variables(variables, reviews, users, votes, users_sim, users_conn):
  """ Populates the dictionary of VariableGroup objects by adding instances
      related to an entity.
  
      Args:
        variables: a dictionary of VariableGroup objects indexed by group names.

      Returns:
        The same dictionary of VariableGroup objects with instances added.
  """
  for vote in votes:
    r_id, a_id, v_id = vote['review'], vote['reviewer'], vote['voter']
    variables['alpha'].add_instance(v_id, map_voter_features(users[v_id]))
    variables['beta'].add_instance(r_id, map_review_features(reviews[r_id]))
    variables['xi'].add_instance(a_id, map_author_features(users[a_id]))
    variables['u'].add_instance(v_id, map_voter_features(users[v_id]))
    variables['v'].add_instance(a_id, map_review_features(reviews[r_id]))
  for author_voter, features in users_sim.items():
    variables['gamma'].add_instance(author_voter, 
        map_users_sim_features(features))
  for author_voter, features in users_conn.items():
    variables['lambda'].add_instance(author_voter,
        map_users_conn_features(features))
  return variables

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
  
 # similar = get_similar_users(users)
 # trusts = parse_trusts()
 # pickle.dump(similar, open('pkl/similar.pkl', 'w'))
 # pickle.dump(trusts, open('pkl/trusts.pkl', 'w'))
  similar = pickle.load(open('pkl/similar.pkl', 'r'))
  trusts = pickle.load(open('pkl/trusts.pkl', 'r'))
  
 # print 'Modeling interaction'
 # sim_author_voter = model_author_voter_similarity(train, users, similar)
 # pickle.dump(sim_author_voter, open('pkl/sim_author_voter.pkl', 'w'))
  sim_author_voter = pickle.load(open('pkl/sim_author_voter.pkl', 'r'))
 # conn_author_voter = model_author_voter_connection(train, users, trusts)
 # pickle.dump(conn_author_voter, open('pkl/conn_author_voter.pkl', 'w'))
  conn_author_voter = pickle.load(open('pkl/conn_author_voter.pkl', 'r'))
  
  print 'Creating variables'
  variables = create_variables()
  populate_variables(variables, reviews, users, sim_author_voter,
      conn_author_voter)
  
  print 'Running EM'
  expectation_maximization(variables, train)

if __name__ == '__main__':
  main() 
