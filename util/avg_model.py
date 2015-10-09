from numpy import isnan, mean

def compute_avg_user(users):
  """ Creates an average user model for mean imputation.

      Args:
        users: the modeled dictionary of users.

      Returns:
        A dictionary representing a user with average feature values.
  """
  avg_user = {}
  for feature in users.itervalues().next():
    avg_user[feature] = 0.0
  missing = []
  for feature in avg_user:
    if feature in ['id', 'ratings', '_id', 'similars', 'trustees', 'trustors']:
      continue
    avg_user[feature] = 0.0
    count = 0
    for user in users.itervalues():
      if not isnan(user[feature]):
        try:
          avg_user[feature] += user[feature]
        except:
          print feature
          print avg_user
          print user
        count += 1
    if count == 0:
      missing.append(feature)
    else:
      avg_user[feature] /= float(count)
  for feature in missing:
    if feature == 'avg_help_giv_sim' or feature == 'avg_help_giv_tru_net':
      avg_user[feature] = avg_user['avg_help_giv']
    elif feature == 'avg_rating_sim' or feature == 'avg_rating_fir_net':
      avg_user[feature] = avg_user['avg_rating']
  avg_user['id'] = avg_user['_id'] = None
  avg_user['ratings'] = {}
  avg_user['similars'] = []
  avg_max = []
  avg_min = []
  for user in users.itervalues():
    if user['ratings']:
      avg_max.append(max(user['ratings'].values()))
      avg_min.append(min(user['ratings'].values()))
  avg_user['avg_max'] = mean(avg_max)
  avg_user['avg_min'] = mean(avg_min)
  return avg_user


def compute_avg_model(model):
  """ Creates an average user model for mean imputation.

      Args:
        users: the modeled dictionary of users.

      Returns:
        A dictionary representing a user with average feature values.
  """
  avg_entity = {}
  for feature in model.itervalues().next():
    avg_entity[feature] = 0.0
  for feature in avg_entity:
    count = 0.0
    for user in model.itervalues():
      if not isnan(user[feature]):
        avg_entity[feature] += user[feature]
        count += 1.0
    avg_entity[feature] /= count
  return avg_entity
