""" Constants Module
    ----------------

    Contains constants shared by multiple algorithms.
"""


NUM_SETS = 5
CONF_QT = 0.975
RANK_SIZE = 5
REP = 3


REVIEW_FEATS = {
  'www': ['num_tokens', 'num_sents', 'uni_ratio', 'avg_sent',
    'cap_sent', 'noun_ratio', 'adj_ratio', 'comp_ratio', 'verb_ratio',
    'adv_ratio', 'fw_ratio', 'sym_ratio', 'num_ratio', 'punct_ratio', 'kl',
    'pos_ratio', 'neg_ratio'],
  'cap': ['num_tokens', 'num_sents', 'uni_ratio', 'avg_sent',
    'cap_sent', 'noun_ratio', 'adj_ratio', 'comp_ratio', 'verb_ratio',
    'adv_ratio', 'fw_ratio', 'sym_ratio', 'num_ratio', 'punct_ratio', 'kl',
    'pos_ratio', 'neg_ratio'],
  'all': ['num_tokens', 'num_sents', 'uni_ratio', 'avg_sent',
    'cap_sent', 'noun_ratio', 'adj_ratio', 'comp_ratio', 'verb_ratio',
    'adv_ratio', 'fw_ratio', 'sym_ratio', 'num_ratio', 'punct_ratio', 'kl',
    'pos_ratio', 'neg_ratio'],
}
AUTHOR_FEATS = {
  'www': ['num_reviews', 'avg_rating', 'num_trustors', 'num_trustees',
    'pagerank'],
  'cap': ['num_reviews', 'avg_rating', 'num_trustors', 'num_trustees',
    'pagerank'],
  'all': ['num_reviews', 'avg_rating', 'num_trustors', 'num_trustees',
    'pagerank', 'num_votes_rec', 'num_votes_giv', 'sd_rating', 'sd_help_rec',
    'sd_help_giv', 'avg_rel_rating', 'avg_help_rec', 'avg_help_giv',
    'avg_rel_help_giv', 'avg_rating_sim', 'avg_help_giv_sim',
    'avg_rating_dir_net', 'avg_help_giv_tru_net']
}
VOTER_FEATS = {
  'www': [],
  'cap': ['num_trustors', 'num_trustees', 'pagerank', 'avg_rating',
    'avg_rating_dir_net', 'avg_rating_sim', 'avg_help_giv', 'avg_help_giv_sim', 
    'avg_help_giv_tru_net'],
  'all': ['num_reviews', 'avg_rating', 'num_trustors', 'num_trustees',
    'pagerank', 'num_votes_rec', 'num_votes_giv', 'sd_rating', 'sd_help_rec',
    'sd_help_giv', 'avg_rel_rating', 'avg_help_rec', 'avg_help_giv',
    'avg_rel_help_giv', 'avg_rating_sim', 'avg_help_giv_sim',
    'avg_rating_dir_net', 'avg_help_giv_tru_net']
}
SIM_FEATS = {
  'www': [],
  'cap': ['common_rated', 'jacc_rated', 'cos_ratings', 'pear_ratings',
    'diff_avg_ratings', 'diff_max_ratings', 'diff_min_ratings'],
  'all': ['common_rated', 'jacc_rated', 'cos_ratings', 'pear_ratings',
    'diff_avg_ratings', 'diff_max_ratings', 'diff_min_ratings']
}
CONN_FEATS = {
  'www': [],
  'cap': ['jacc_trustees', 'jacc_trustors', 'adamic_adar_trustees',
    'adamic_adar_trustors', 'katz'],
  'all': ['jacc_trustees', 'jacc_trustors', 'adamic_adar_trustees',
    'adamic_adar_trustors', 'katz']
}
