from re import match

from networkx import DiGraph
from numpy import mean, nan
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet

from preprocessing.parsing import parse_votes

_FILE = 'data/rating.txt'
_NEWFILE = 'data/reviews.txt'


def get_foreign_ratio(text):
  """ Get ratio of foreign (or unidentified) words in a text.

      Args:
        text: a string containing the text.

      Returns:
        A real value with the ratio of foreign words.
  """
  text = text.lower()
  word_tokenizer = RegexpTokenizer(r'\w+')
  words = word_tokenizer.tokenize(text)
  fw = [w for w in words if match('[a-z]+', w) and wordnet.synsets(w) == []]
  if len(words) > 0:  
    return float(len(fw)) / len(words)
  else:
    return 0

f = open(_FILE, 'r')
output = open(_NEWFILE, 'w')
review_count = 0
count_lines = 0
count_ignored = 0
type_ignored = {'user': 0, 'product': 0, 'category': 0, 'rating': 0,
    'date': 0, 'text': 0, 'votes': 0}
for line in f:
  count_lines += 1
  try:
    l = line.strip().split('::::')
    review = {}
    review['id'] = review_count
    review_count += 1
    if not l[0].strip():
      raise Exception('author')
    if not l[1].strip():
      raise Exception('product')
    if not l[2].strip():
      raise Exception('category')
    try:
      rating = int(l[3]) / 10
      if rating < 0 or rating > 5:
        raise Exception()
    except Exception:
      raise Exception('rating')
    if not l[5].strip():
      raise Exception('date')
    if not l[6].strip() or get_foreign_ratio(l[6].strip()) >= 0.4:
      raise Exception('text')
    try:
      review['votes'] = parse_votes(l[7])
    except Exception:
      raise Exception('votes')
    if not review['votes']:
      raise Exception('votes')
  except Exception as e:
    print 'Exception on parsing review, line %d, type %s' % (count_lines,
        e.args[0])
    print l
    print '--------------------------'
    count_ignored += 1
    if e.args[0] in type_ignored:
      type_ignored[e.args[0]] += 1
    continue
  print >> output, line 

print '#############################'
print 'Summary of Filtering:'
print '~ Ignored: %d' % count_ignored
for item in type_ignored.items():
  print '~ Ignored of type %s: %d' % item
print '#############################'

f.close()
output.close()
