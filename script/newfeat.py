""" Script for generating new features:
    New proposed interaction features are generated, and new pickles for
    similarity and connection are provided with new-prefix.
"""

from pickle import load, dump

from numpy import zeros, mean, array, nan
from networkx import single_source_shortest_path_length, closeness_centrality, \
    betweenness_centrality, eigenvector_centrality

from util.aux import cosine
from util.avg_model import compute_avg_user

_OUTPUT_DIR = 'data/'
_PKL_DIR = 'out/pkl'

def main():
  """ Models the whole dataset using features and output to a file. 

      Args:
        None.

      Returns:
        None.
  """
  for i in xrange(1, 5):
    print 'Reading data'
    trusts = load(open('%s/trusts.pkl' % _PKL_DIR, 'r'))
    reviews = load(open('%s/reviews-%d.pkl' % (_PKL_DIR, i), 'r'))
    users = load(open('%s/users-%d.pkl' % (_PKL_DIR, i), 'r'))
    train = load(open('%s/train-%d.pkl' % (_PKL_DIR, i), 'r'))
    validation = load(open('%s/validation-%d.pkl' % (_PKL_DIR, i), 'r'))
    test = load(open('%s/test-%d.pkl' % (_PKL_DIR, i), 'r'))
    sim = load(open('%s/sim-%d.pkl' % (_PKL_DIR, i), 'r'))
    conn = load(open('%s/conn-%d.pkl' % (_PKL_DIR, i), 'r'))

    print 'Generating similarity'
    avg_user = compute_avg_user(users)
    close = {}
    eigen = eigenvector_centrality(trusts) 
    for author, voter in sim:
      author_dic = users[author] if author in users else avg_user
      voter_dic = users[voter] if voter in users else avg_user
      # if any feature is nan, the derivated becomes nan and will be imputated
      sim[(author, voter)]['diff_trustors'] = author_dic['num_trustors'] - \
          voter_dic['num_trustors'] 
      sim[(author, voter)]['diff_reviews'] = author_dic['num_reviews'] - \
          voter_dic['num_reviews'] 
      sim[(author, voter)]['diff_pagerank'] = author_dic['pagerank'] - \
          voter_dic['pagerank']
      if voter not in close:
        close[voter] = closeness_centrality(trusts, voter) if voter in trusts \
            else nan
      if author not in close:
        close[author] = closeness_centrality(trusts, author) if author in trusts\
            else nan
      sim[(author, voter)]['diff_close'] = close[author] - close[voter] 
      if voter not in eigen:
        eigen[voter] = nan
      if author not in eigen:
        eigen[author] = nan
      conn[(author, voter)]['diff_eigen'] = eigen[author] - eigen[voter] 
    dump(sim, open('%s/new-sim-%d.pkl' % (_PKL_DIR, i), 'w'))

    print 'Generating connection'
    paths = {}
    for author, voter in conn:
      conn[(author, voter)]['voter_trust'] = 1 if \
          trusts.has_edge(voter, author) else 0
      conn[(author, voter)]['author_trust'] = 1 if \
          trusts.has_edge(author, voter) else 0
      if voter not in paths and voter in trusts:
        paths[voter] = single_source_shortest_path_length(trusts, voter) 
      if author not in paths and author in trusts:
        paths[author] = single_source_shortest_path_length(trusts, author)
      conn[(author, voter)]['inv_from_vot_path'] = 0 if voter not in trusts \
          or author not in paths[voter] else (1.0 / float(paths[voter][author]))
      conn[(author, voter)]['inv_from_aut_path'] = 0 if author not in trusts \
          or voter not in paths[author] else (1.0 / float(paths[author][voter]))
    dump(conn, open('%s/new-conn-%d.pkl' % (_PKL_DIR, i), 'w')) 

if __name__ == '__main__':
  main()
