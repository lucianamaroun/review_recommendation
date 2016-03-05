from re import match

from networkx import DiGraph
from numpy import mean, nan
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet

from prep.parsing import parse_votes

_FILE = 'data/reviews.txt'
_NEWFILE = 'data/reviews_filtered.txt'


f = open(_FILE, 'r')
user_votes = {}
for line in f:
  l = line.strip().split('::::')
  votes = parse_votes(l[7], l[0].strip())
  product = l[1].strip()
  for voter in votes:
    if (product, voter) not in user_votes:
      user_votes[(product, voter)] = 0
    user_votes[(product, voter)] += 1

sel_keys = set()
for key in user_votes:
  if user_votes[key] >= 10:
    sel_keys.add(key)

f.seek(0, 0)
output = open(_NEWFILE, 'w')
for line in f:
  l = line.strip().split('::::')
  str_votes = l[7].split(':::') 
  new_str_votes = []
  product = l[1].strip()
  for str_vote in str_votes:
    voter = str_vote.split(':')[0]
    if (product, voter) in sel_keys:
      new_str_votes.append(str_vote)
  if len(new_str_votes) <= 0:
    # no vote of this review are included, ignore review
    continue
  new_str_votes.append('</endperson>')
  l[7] = ':::'.join(new_str_votes)
  print >> output, '::::'.join(l)

f.close()
output.close()
