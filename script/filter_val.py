from pickle import load, dump

for i in xrange(5):
  reviews = load(open('out/pkl/reviews-%d.pkl' % i, 'r'))
  val = load(open('out/pkl/validation-%d.pkl' % i, 'r'))
  pair_votes = {}
  for vote in val:
    voter = vote['voter']
    product = reviews[vote['review']]['product']
    if (product, voter) not in pair_votes:
      pair_votes[(product, voter)] = 0
    pair_votes[(product, voter)] += 1
  sel_keys = set()
  for key in pair_votes:
    if pair_votes[key] >= 5:
      sel_keys.add(key)
  new_val = []
  for vote in val:
    voter = vote['voter']
    product = reviews[vote['review']]['product']
    if (product, voter) in sel_keys:
      new_val.append(vote)
  print 'Size of old val #%d: %d' % (i + 1, len(val))
  print 'Size of new val #%d: %d' % (i + 1, len(new_val))
  dump(new_val, open('out/pkl/filt_val-%d.pkl' % i, 'w'))
