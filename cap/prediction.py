""" Prediction Module for CAP Baseline
    ----------------------------------

    This module implements a Monte Carlo EM algorithm for fitting latent
    variables and data distributions' parameters.

    Usage:
      $ python -m cap.prediction
    on root directory of the project.
"""

import numpy as np
from math import sqrt

from src.modeling import model
from src.author_voter_modeling import model_author_voter_similarity, \
    model_author_voter_connection
from src.user_modeling import get_similar_users
from src.parser import parse_trusts
from cap.models import EntityScalarGroup, EntityArrayGroup, \
    InteractionScalarGroup, EntityScalarParameter, EntityArrayParameter, \
    InteractionScalarParameter, ScalarVarianceParameter, \
    ArrayVarianceParameter, PredictionVarianceParameter
from cap import const
from cap.em import expectation_maximization
from cap.map_features import map_review_features, map_author_features, \
    map_voter_features, map_users_sim_features, map_users_conn_features


""" Creates empty latent variable groups with its corresponding parameters.

    Args:
      None.
    
    Returns:
      A dictionary of Group objects indexed by names.
"""
def create_variables():
  var_H = PredictionVarianceParameter('var_H')
  variables = {
    'alpha': EntityScalarGroup('alpha', 'voter', 
        EntityScalarParameter('d', (9, 1)), 
        ScalarVarianceParameter('var_alpha'), var_H),
    'beta': EntityScalarGroup('beta', 'review', 
        EntityScalarParameter('g', (17, 1)), 
        ScalarVarianceParameter('var_beta'), var_H),
    'xi': EntityScalarGroup('xi', 'reviewer', 
        EntityScalarParameter('b', (5, 1)), 
        ScalarVarianceParameter('var_xi'), var_H),
    'u': EntityArrayGroup('u', (const.K, 1), 'voter', 
        EntityArrayParameter('W', (const.K, 9)),
        ArrayVarianceParameter('var_u'), var_H),
    'v': EntityArrayGroup('v', (const.K, 1), 'review', 
        EntityArrayParameter('V', (const.K, 17)),
        ArrayVarianceParameter('var_v'), var_H),
    'gamma': InteractionScalarGroup('gamma', ('reviewer', 'voter'),
        InteractionScalarParameter('r', (7, 1)), 
        ScalarVarianceParameter('var_gamma'), var_H),
    'lambda': InteractionScalarGroup('lambda', ('reviewer', 'voter'),
        InteractionScalarParameter('h', (4, 1)), 
        ScalarVarianceParameter('var_lambda'), var_H)
  }
  variables['u'].set_pair_name('v')
  variables['v'].set_pair_name('u')
  return variables


def populate_variables(variables, reviews, users, votes, users_sim, users_conn):
  """ Populates the dictionary of Group objects by adding instances
      related to an entity.
  
      Args:
        variables: a dictionary of Group objects indexed by group names.
        reviews: dictionary of reviews' features indexed by id.
        authors: dictionary of authors' features indexed by id.
        votes: dictionary of votes' features indexed by id.
        users_sim: dictionary of users similarity features indexed by a tuple
          (author_id, voter_id)
        users_conn: dictionary of users connection features indexed by a tuple
          (author_id, voter_id)

      Returns:
        The same dictionary of Group objects with instances added.
  """
  for vote in votes:
    r_id, a_id, v_id = vote['review'], vote['reviewer'], vote['voter']
    variables['alpha'].add_instance(v_id, map_voter_features(users[v_id]))
    variables['beta'].add_instance(r_id, map_review_features(reviews[r_id]))
    variables['xi'].add_instance(a_id, map_author_features(users[a_id]))
    variables['u'].add_instance(v_id, map_voter_features(users[v_id]))
    variables['v'].add_instance(r_id, map_review_features(reviews[r_id]))
  for author_voter, features in users_sim.items():
    variables['gamma'].add_instance(author_voter, 
        map_users_sim_features(features))
  for author_voter, features in users_conn.items():
    variables['lambda'].add_instance(author_voter,
        map_users_conn_features(features))
  return variables


def calculate_predictions(groups, test, reviews, users, users_sim, users_conn):
  pred = []
  ignored = 0
  for vote in test:
    r_id, a_id, v_id = vote['review'], vote['reviewer'], vote['voter']
    alfa = beta = 0
    u = v = np.zeros((const.K, 1))
    if v_id in users:
      alfa = groups['alpha'].get_instance(vote).value if \
          groups['alpha'].contains(vote) else groups['alpha'].weight_param.value.T \
          .dot(map_voter_features(users[v_id]))[0,0]
      u = groups['u'].get_instance(vote).value if groups['u'].contains(vote) \
          else groups['u'].weight_param.value \
          .dot(map_voter_features(users[v_id]))
    if r_id in reviews:
      beta = groups['beta'].get_instance(vote).value if \
          groups['beta'].contains(vote) else groups['beta'].weight_param.value.T \
          .dot(map_review_features(reviews[r_id]))[0,0]
      v = groups['v'].get_instance(vote).value if groups['v'].contains(vote) \
          else groups['v'].weight_param.value \
          .dot(map_review_features(reviews[r_id]))
    if a_id in users:
      xi = groups['xi'].get_instance(vote).value if \
          groups['xi'].contains(vote) else groups['xi'].weight_param.value.T \
          .dot(map_author_features(users[a_id]))[0,0]
    gamma = groups['gamma'].get_instance(vote).value if \
        groups['gamma'].contains(vote) else 0
    if not gamma and (a_id, v_id) in users_sim:
      gamma = groups['gamma'].weight_param.value.T \
      .dot(map_users_sim_features(users_sim[(a_id, v_id)]))[0,0]
    lambd = 0
   # lambd = groups['lambda'].get_instance(vote).value if \
   #     groups['lambda'].contains(vote) else 0
   # if not lambd and (a_id, v_id) in users_conn:
   #   lambd = groups['lambda'].weight_param.value.T \
   #   .dot(map_users_conn_features(users_conn[(a_id, v_id)]))[0,0]
    prediction = u.T.dot(v)[0,0] + alfa + beta + xi + gamma + lambd
    pred.append(prediction)
  return pred


def main():
  import pickle
  print 'Reading pickles'
 # reviews, users, _, train, test = model()
 # pickle.dump(reviews, open('pkl/cap_reviews2.pkl', 'w'))
 # pickle.dump(users, open('pkl/cap_users2.pkl', 'w'))
 # pickle.dump(train, open('pkl/cap_train2.pkl', 'w'))
 # pickle.dump(test, open('pkl/cap_test2.pkl', 'w'))
  reviews = pickle.load(open('pkl/cap_reviews2.pkl', 'r'))
  users = pickle.load(open('pkl/cap_users2.pkl', 'r'))
  train = pickle.load(open('pkl/cap_train2.pkl', 'r'))
  test = pickle.load(open('pkl/cap_test2.pkl', 'r'))
  
 # similar = get_similar_users(users)
 # trusts = parse_trusts()
 # pickle.dump(similar, open('pkl/cap_similar2.pkl', 'w'))
 # pickle.dump(trusts, open('pkl/cap_trusts2.pkl', 'w'))
  similar = pickle.load(open('pkl/cap_similar2.pkl', 'r'))
  trusts = pickle.load(open('pkl/cap_trusts2.pkl', 'r'))
  
  print 'Modeling interaction'
 # sim_author_voter = model_author_voter_similarity(train, users, similar)
 # pickle.dump(sim_author_voter, open('pkl/cap_sim_author_voter22.pkl', 'w'))
  sim_author_voter = pickle.load(open('pkl/cap_sim_author_voter22.pkl', 'r'))
 # conn_author_voter = model_author_voter_connection(train, users, trusts)
 # pickle.dump(conn_author_voter, open('pkl/cap_conn_author_voter22.pkl', 'w'))
  conn_author_voter = pickle.load(open('pkl/cap_conn_author_voter22.pkl', 'r'))
  
  print 'Creating variables'
  variables = create_variables()
  populate_variables(variables, reviews, users, train, sim_author_voter,
      conn_author_voter)
  
  print 'Running EM'
  expectation_maximization(variables, train)
  pickle.dump(variables, open('pkl/cap_variables.pkl', 'w'))
  #variables = pickle.load(open('pkl/cap_variables.pkl', 'r'))
  #train_truth = [t['vote'] for t in train]
  #overall_avg = float(sum(train_truth)) / len(train_truth)

  print 'Calculate Predictions'
  pred = calculate_predictions(variables, test, reviews, users, sim_author_voter,
    conn_author_voter)
  sse = sum([(pred[i] - test[i]['vote']) ** 2 for i in xrange(len(test))])
  rmse = sqrt(sse/len(test))

  print 'RMSE: %s' % rmse

if __name__ == '__main__':
  main() 
