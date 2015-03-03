""" LDA Script
    ----------

    Applies LDA to a subset of the reviews and prints the topics to stdout. For
    a different number of reviews, topics or idf filter quantile, change
    variables _N_REVIEWS, _N_TOPICS and _IDF_QUANTILE, respectively.

    Usage:
      $ python -m script.lda
    on the root directory of the project.
"""


from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from gensim import corpora, models
from math import log, floor
from textblob import TextBlob

from src import parser


_N_REVIEWS = float('inf') # 30454
_N_TOPICS = 5
_COMMON_THR = 0.8
_RARE_THR = 0.1
_CATEGORY = None 
_SUBSTITUTE = {
  # unicode as symbol
    '  \d\d\d\d;': '#', '&#\d\d\d\d;': '#', 'quot;': '#', 'lt;': '#',
    'gt;': '#', 'quot': '#', '8217': '#', '8221': '#', '8220': '#', '8211': '#',
  # fixing negation contraction
    'dont': 'don\'t', 'didnt': 'didn\'t', 'havent': 'haven\'t',
    'wouldnt': 'wouldn\'t', 'wont': 'won\'t', 'doesnt': 'doesn\'t',
    'isnt': 'isn\'t', 'wasnt': 'wasn\'t', 'shouldnt': 'shouldn\'t',
    'couldnt': 'couldn\'t', 'cant': 'can\'t', 'werent': 'weren\'t'
  }


""" Extracts words from text, ignoring pucntuation.

    Args:
      text: a string.

    Returns:
      A list of strings with words extracted in lower case.
"""
def extract_words(text):
    word_tokenizer = RegexpTokenizer(r'\w+')
    words = word_tokenizer.tokenize(text)
    return [w.lower() for w in words]


""" Removes stopwords.

    Args:
      words: a list of strings.

    Returns:
      A list of strings with stopwords removed.
"""
def remove_stopwords(words):
  stopw = set(stopwords.words('english'))
  words = [w for w in words if w not in stopw]
  return words

""" Stems a list of words.

    Args:
      words: a list of strings to stem.

    Returns:
      A list of strings with stemmed words.
"""
def stem_list(words):
  stemmer = SnowballStemmer('english')
  return [stemmer.stem(w) for w in words]


""" Performs a series of transformation of documents, extracting words,
    removing stopwords and stemming.

    Args:
      documents: a list of strings representing documents.

    Returns:
      A list of list of strings with each document represented by a list of
    stemmed and filtered words.
"""
def process_documents(documents):
  documents = [extract_words(d) for d in documents]
  documents = [remove_stopwords(d) for d in documents]
  #documents = remove_rare_words(documents)
  #documents = remove_common_words(documents)
  #documents = [stem_list(d) for d in documents]
  return documents


""" Removes rare words, which have document frequency lower than a threshold.

    Args:
      documents: a list of list of strings representing documents.

    Returns:
      A list of list of strings (i.e. a list of documents).
"""
def remove_rare_words(documents):
  dfs = compute_dfs(documents)
  return [[w for w in d if dfs[w] >= _RARE_THR] for d in documents]


""" Removes common words, which have document frequency higher than a threshold.

    Args:
      documents: a list of list of strings representing documents.

    Returns:
      A list of list of strings (i.e., a list of documents).
"""
def remove_common_words(documents):
  dfs = compute_dfs(documents)
  return [[w for w in d if dfs[w] <= _COMMON_THR] for d in documents]


""" Computes document frequencies from a collection of documents.

    Args:
      documents: a list of list of strings representing documents.

    Returns:
      A dictionary indexed by words containing the df of the correspoding word.
"""
def compute_dfs(documents):
  dfs = {}
  for d in documents:
    for w in set(d):
      if w not in dfs:
        dfs[w] = 0
      dfs[w] += 1
  for w in dfs:
    dfs[w] = dfs[w] / len(documents)
  return dfs


if __name__ == '__main__':
  count = 0
  documents = []
  for r in parser.parse_reviews():
    if _CATEGORY and r['category'] != _CATEGORY:
      continue
    text = r['text']
    text = text.lower()
    for sub, target in _SUBSTITUTE.items():
      text = text.replace(sub, target)
    documents.append(text)
    count += 1
    if count >= _N_REVIEWS:
      break
  documents = process_documents(documents)
  print len(documents)
  dictionary = corpora.Dictionary(documents)
  print len(dictionary)
  corpus = [dictionary.doc2bow(d) for d in documents]
  print len(corpus)
  lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary,
      num_topics=_N_TOPICS)
  print lda
  topics = lda.print_topics(_N_TOPICS)
  for index, topic in enumerate(topics):
    print 'Topic %d' % (index + 1)
    print topic
    print '-----------------'
