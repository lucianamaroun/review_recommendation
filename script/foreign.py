""" Foreign Script
    --------------

    Evaluates presence of foreign (or unidentified) words in reviews. The
    result is output to stdout containing the reviews with 40% of foreign words
    or more and a summary with the total of foreign reviews identified.

    Usage:
      $ python -m script.foreign
    on the root directory of the project.
"""

from re import match

from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer

from src import parser

count_foreign = 0
max_ratio = 0
max_text = None
for r in parser.parse_reviews():
  text = r['text'].lower()
  word_tokenizer = RegexpTokenizer(r'\w+')
  words = word_tokenizer.tokenize(text)
  fw = [w for w in words if match('[a-z]+', w) and wordnet.synsets(w) == []]
  try:
    fw_ratio = float(len(fw)) / len(words)
  except Exception:
    # print text
    continue
  if fw_ratio >= 0.4:
    print 'FW Review found: %d' % r['id']
    print text
    print '-----------------------------'
    count_foreign += 1
  if fw_ratio > max_ratio:
    max_ratio = fw_ratio
    max_text = text
print '******************************'
print 'SUMMARY'
print '# fw reviews: %d' % count_foreign
print '******************************'
