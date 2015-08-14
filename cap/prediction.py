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

from src.modeling.modeling import model, _SAMPLE_RATIO
from src.modeling.author_voter_modeling import model_author_voter_similarity, \
    model_author_voter_connection
from src.modeling.user_modeling import get_similar_users
from src.parsing.parser import parse_trusts
from cap.models import EntityScalarGroup, EntityArrayGroup, \
    InteractionScalarGroup, EntityScalarParameter, EntityArrayParameter, \
    InteractionScalarParameter, ScalarVarianceParameter, \
    ArrayVarianceParameter, PredictionVarianceParameter
from cap import const
from cap.em import expectation_maximization
from cap.map_features import map_review_features, map_author_features, \
    map_voter_features, map_users_sim_features, map_users_conn_features


def create_variables():
  """ Creates empty latent variable groups with its corresponding parameters.

      Args:
        None.
      
      Returns:
        A dictionary of Group objects indexed by names.
  """
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
        InteractionScalarParameter('h', (5, 1)), 
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
  """ Calculate the predictions after fitting values. If the vote to be
      predicted contains entities modeled as latent variables (i.e., present
      on training set), the latent variable is used; otherwise, it is
      approximated by linear regression over features.

      Args:
        groups: dictionary of Group objects.
        test: list of vote dictionaries on test set.
        reviews: dictionary of review dictionaries.
        users: dictionary of user dictionaries.
        users_sim: dictionary of similarity of users dictionaries.
        users_conn: dictionary of connection of users dictionaries.

      Returns:
        A list of floats containing prediction values for each vote in test, in
      the same order.
  """
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
    lambd = groups['lambda'].get_instance(vote).value if \
        groups['lambda'].contains(vote) else 0
    if not lambd and (a_id, v_id) in users_conn:
      lambd = groups['lambda'].weight_param.value.T \
      .dot(map_users_conn_features(users_conn[(a_id, v_id)]))[0,0]
    prediction = u.T.dot(v)[0,0] + alfa + beta + xi + gamma + lambd
    pred.append(prediction)
  return pred


def main():
  import pickle
  print 'Reading pickles'
 # reviews, users, _, train, test = model()
 # pickle.dump(reviews, open('pkl/cap_reviews%f.pkl' % _SAMPLE_RATIO, 'w'))
 # pickle.dump(users, open('pkl/cap_users%f.pkl' % _SAMPLE_RATIO, 'w'))
 # pickle.dump(train, open('pkl/cap_train%f.pkl' % _SAMPLE_RATIO, 'w'))
 # pickle.dump(test, open('pkl/cap_test%f.pkl' % _SAMPLE_RATIO, 'w'))
  reviews = pickle.load(open('pkl/cap_reviews%f.pkl' % _SAMPLE_RATIO, 'r'))
  users = pickle.load(open('pkl/cap_users%f.pkl' % _SAMPLE_RATIO, 'r'))
  train = pickle.load(open('pkl/cap_train%f.pkl' % _SAMPLE_RATIO, 'r'))
  test = pickle.load(open('pkl/cap_test%f.pkl' % _SAMPLE_RATIO, 'r'))
  
 # similar = get_similar_users(users)
 # trusts = parse_trusts()
 # pickle.dump(similar, open('pkl/cap_similar%f.pkl' % _SAMPLE_RATIO, 'w'))
 # pickle.dump(trusts, open('pkl/cap_trusts%f.pkl' % _SAMPLE_RATIO, 'w'))
  similar = pickle.load(open('pkl/cap_similar%f.pkl' % _SAMPLE_RATIO, 'r'))
  trusts = pickle.load(open('pkl/cap_trusts%f.pkl' % _SAMPLE_RATIO, 'r'))
  
  print 'Modeling interaction'
 # sim_author_voter = model_author_voter_similarity(train, users, similar)
 # pickle.dump(sim_author_voter, open('pkl/cap_sim_author_voter%f.pkl' % _SAMPLE_RATIO, 'w'))
  sim_author_voter = pickle.load(open('pkl/cap_sim_author_voter%f.pkl' % _SAMPLE_RATIO, 'r'))
 # conn_author_voter = model_author_voter_connection(train, users, trusts)
 # pickle.dump(conn_author_voter, open('pkl/cap_conn_author_voter%f.pkl' % _SAMPLE_RATIO, 'w'))
  conn_author_voter = pickle.load(open('pkl/cap_conn_author_voter%f.pkl' % _SAMPLE_RATIO, 'r'))
  
  print 'Creating variables'
  variables = create_variables()
  populate_variables(variables, reviews, users, train, sim_author_voter,
      conn_author_voter)
  
  print 'Running EM'
  expectation_maximization(variables, train)
  pickle.dump(variables, open('pkl/cap_variables%f.pkl' % _SAMPLE_RATIO, 'w'))
  #variables = pickle.load(open('pkl/cap_variables.pkl', 'r'))
  #train_truth = [t['vote'] for t in train]
  #overall_avg = float(sum(train_truth)) / len(train_truth)

  print 'Calculate Predictions'
  pred = calculate_predictions(variables, train, reviews, users, sim_author_voter,
    conn_author_voter)
   
  print "TRAINING ERROR"
  sse = sum([(pred[i] - train[i]['vote']) ** 2 for i in xrange(len(train))])
  rmse = sqrt(sse/len(train))
  print 'RMSE: %s' % rmse
  
  pred = calculate_predictions(variables, test, reviews, users, sim_author_voter,
    conn_author_voter)
   
  print "TESTING ERROR"
  sse = sum([(pred[i] - test[i]['vote']) ** 2 for i in xrange(len(test))])
  rmse = sqrt(sse/len(test))
  print 'RMSE: %s' % rmse


if __name__ == '__main__':
  main() 
