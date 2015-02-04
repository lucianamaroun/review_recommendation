from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from gensim import corpora, models
from math import log, floor

from src import parser


_IDF_QUANTILE = 0.01


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
  documents = remove_common_words(documents)
  documents = [stem_list(d) for d in documents]
  return documents


""" Removes common words using an idf filter.

    Args:
      documents: a list of list of strings representing documents.

    Returns:
      A list of list of strings.
"""
def remove_common_words(documents):
  idfs = compute_idfs(documents)
  thr = sorted(idfs.values())[int(len(idfs) * _IDF_QUANTILE)]
  return [[w for w in d if idfs[w] >= thr] for d in documents]


""" Computes IDFs from a collection of documents.

    Args:
      documents: a list of list of strings representing documents.

    Returns:
      A dictionary indexed by words containing the idf of the correspoding word.
"""
def compute_idfs(documents):
  idfs = {}
  for d in documents:
    for w in set(d):
      if w not in idfs:
        idfs[w] = 0
      idfs[w] += 1
  for w in idfs:
    idfs[w] = log(len(documents) / float(idfs[w]))
  return idfs


if __name__ == '__main__':
  count = 0
  documents = []
  for r in parser.parse_reviews():
    documents.append(r['text'])
    count += 1
    if count >= 10000:
      break
  documents = process_documents(documents)
  dictionary = corpora.Dictionary(documents)
  corpus = [dictionary.doc2bow(d) for d in documents]
  lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary,
      num_topics=10)
  topics = lda.print_topics(10)
  for index, topic in enumerate(topics):
    print 'Topic %d' % (index + 1)
    print topic
    print '-----------------'
