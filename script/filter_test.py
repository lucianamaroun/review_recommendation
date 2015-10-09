from pickle import load, dump

for i in xrange(5):
  reviews = load(open('out/pkl/reviews-%d.pkl' % i, 'r'))
  test = load(open('out/pkl/test-%d.pkl' % i, 'r'))
  pair_votes = {}
  for vote in test:
    voter = vote['voter']
    product = reviews[vote['review']]['product']
    if (product, voter) not in pair_votes:
      pair_votes[(product, voter)] = 0
    pair_votes[(product, voter)] += 1
  sel_keys = set()
  for key in pair_votes:
    if pair_votes[key] >= 5:
      sel_keys.add(key)
  new_test = []
  for vote in test:
    voter = vote['voter']
    product = reviews[vote['review']]['product']
    if (product, voter) in sel_keys:
      new_test.append(vote)
  print 'Size of old test #%d: %d' % (i + 1, len(test))
  print 'Size of new test #%d: %d' % (i + 1, len(new_test))
  dump(new_test, open('out/pkl/filt_test-%d.pkl' % i, 'w'))
