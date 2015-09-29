""" Review modeling module
    ----------------------

    Models reviews with a set of features derived from rating and text.

    Usage:
      Used only as a module, not directly callable.
"""


from string import punctuation
from math import log, ceil
from copy import deepcopy
from multiprocessing import Pool
from re import match
from traceback import print_exc

from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk import pos_tag
from nltk.corpus import wordnet
from pymongo import MongoClient
from numpy import nan, isnan, mean

from preprocessing.parsing import parse_reviews
from lib.sentiment.sentiwordnet import SimplifiedSentiWordNet


_NEGATION = set(['not', 'no', 'n\'t'])
_USE_DB = False
_SUBSTITUTE = {
  # unicode as symbol
    '  \d\d\d\d;': '#', '&#\d\d\d\d;': '#', 'quot;': '#', 'lt;': '#',
    'gt;': '#',
  # fixing negation contraction
    'dont': 'don\'t', 'didnt': 'didn\'t', 'havent': 'haven\'t',
    'wouldnt': 'wouldn\'t', 'wont': 'won\'t', 'doesnt': 'doesn\'t',
    'isnt': 'isn\'t', 'wasnt': 'wasn\'t', 'shouldnt': 'shouldn\'t',
    'couldnt': 'couldn\'t', 'cant': 'can\'t', 'werent': 'weren\'t'
  }
_PUNCTUATION = set(['!', '?', ':', ';', ',', '.']) 
    # source: http://www.nltk.org/api/nltk.tokenize.html
_SYMBOLS = punctuation

if _USE_DB:
  client = MongoClient('mongodb://localhost:27017/')
  reviews_db = client.review_rec.reviews
_SWN = SimplifiedSentiWordNet()


def model_review_text(text_pair):
  """ Models a review.

      Args:
        text_pair: a pair with review id and review text. 

      Returns:
        A pair with review id and a dictionary with keys represented text 
      related modeled attributes.
  """
  try:
    r_id, text = text_pair
    review = get_textual_features(text)
    if _USE_DB:
      review['_id'] = r_id
      reviews_db.insert_one(review)
    else:
      return r_id, review
  except Exception as e:
    print_exc()
    print ''
    raise e


def get_textual_features(text):
  """ Reports different features derived from text, related to length, syntax,
      lexicon and sentiment statistics.

      Args:
        text: a string with the text to derive the features from.

      Returns:
        A dictionary from feature names to feature value. Refer to
      "get_text_length_stats", "get_pos_stats" and "get_sent_stats" for complete
      list of feature names.
  """
  lower_text = text.lower()
  for sub, target in _SUBSTITUTE.items():
    lower_text = lower_text.replace(sub, target)

  features = {}
  length_feat = get_text_length_stats(text)
  for feat in length_feat:
    features[feat] = length_feat[feat]

  tokens = word_tokenize(lower_text)
  tags = pos_tag(tokens)
  pos_feat = get_pos_stats(tags)
  for feat in pos_feat:
    features[feat] = pos_feat[feat]

  sent_feat = get_sent_stats(tags)
  for feat in sent_feat:
    features[feat] = sent_feat[feat]

  return features


def get_text_length_stats(text):
  """ Gets length statistics of the text: number of chars, number of tokens,
      number of words, number of sentences, ratio of unique words, average number
      of tokens by sentence and ratio of capitalized sentences.

      Args:
        text: a string with the text to calculate statistics from.

      Returns:
        A dictionary indexed by keys "num_chars", "num_tokens", "num_words",
      "num_sents", "uni_ratio", "avg_sent" and "cap_sent", and containing as
      values the corresponding statistic value.
  """
  features = {}

  word_tokenizer = RegexpTokenizer(r'\w+')

  tokens = word_tokenize(text)
  words = word_tokenizer.tokenize(text)
  sents = sent_tokenize(text)
  capsents = [s for s in sents if match('[A-Z]', s[0])] 

  features['num_chars'] = len(text)
  features['num_tokens'] = len(tokens)
  features['num_words'] = len(words)
  features['num_sents'] = len(sents)
  features['uni_ratio'] = float(len(set(words))) / len(words) if len(words) \
      else 0.0
  features['avg_sent'] = float(features['num_tokens']) / \
      features['num_sents'] if features['num_sents'] else 0.0
  features['cap_sent'] = float(len(capsents)) / len(sents) if len(sents) \
      else 0.0

  return features


def get_pos_stats(tags):
  """ Gets ratios of some POS tags: noun, adjective, adverb, comparatives
      (adjective or adverb), foreign words, symbols, numbers and punctuations.

      Args:
        tags: list of pairs (word, tag). 

      Returns:
        A dictionary indexed by keys "noun_ratio", "adj_ratio", "adv_ratio",
      "verb_ratio", "comp_ratio", "fw_ratio", "sym_ratio", "num_ratio" and
      "punct_ratio", and containing as values the ratios of the corresponding
      tag accross all words of the text.
  """
  features = {'noun_ratio': 0.0, 'adj_ratio': 0.0, 'adv_ratio': 0.0,
      'verb_ratio': 0.0, 'comp_ratio': 0.0, 'fw_ratio': 0.0,
      'sym_ratio': 0.0, 'num_ratio': 0.0, 'punct_ratio': 0.0}

  for word, tag in tags:
    if tag == 'FW' or (match('[a-z]+', tag) and wordnet.synsets(tag) == []):
      features['fw_ratio'] += 1.0 # order is important
    elif tag.startswith('NN'):
      features['noun_ratio'] += 1.0
    elif tag.startswith('JJ'):
      features['adj_ratio'] += 1.0
    elif tag.startswith('RB'):
      features['adv_ratio'] += 1.0
    elif tag.startswith('VB'):
      features['verb_ratio'] += 1.0
    elif tag == 'JJR' or tag == 'RBR':
      features['comp_ratio'] += 1.0
    elif tag == 'CD':
      features['num_ratio'] += 1.0
    elif tag == 'SYM' or match('[' + _SYMBOLS + ']+', word):
      features['sym_ratio'] += 1.0
    elif tag == 'SYM' or tag in _PUNCTUATION:
      features['punct_ratio'] += 1.0

  for tag in ['noun_ratio', 'adj_ratio', 'adv_ratio', 'verb_ratio', 'comp_ratio',
          'fw_ratio', 'sym_ratio', 'num_ratio', 'punct_ratio']:
      features[tag] = features[tag] / len(tags) if len(tags) else 0.0

  return features


def get_sent_stats(tags):
  """ Gets positive and negative ratio of sentences and words. Sentence ratio
      has a better context for deriving meaning and, consequently, sentiment. Word
      related ratio is calculated using the first meaning of the identified tag.

      Args:
        tags: list of pairs (word, tag). 

      Returns:
        Two real values, pos_ratio and neg_ratio, with, respectively the ratio of
      positive sentences and the ratio of negative sentences.
  """
  features = {}
  negate = False

  features['pos_ratio'] = features['neg_ratio'] = w_count = 0.0
  for word, tag in tags:
    if tag == 'SYM' or tag in _PUNCTUATION:
      negate = False
      continue
    if word in _NEGATION:
      negate = True
    w_count += 1.0
    if tag.startswith('NN'):
      tag = 'n'
    elif tag.startswith('JJ'):
      tag = 'a'
    elif tag.startswith('VB'):
      tag = 'v'
    elif tag.startswith('RB'):
      tag = 'r'
    else:
      continue
    scores = _SWN.scores(word, tag)
    if scores:
      if negate:
        scores['pos_score'], scores['neg_score'] = scores['neg_score'], \
            scores['pos_score']
      if scores['pos_score'] > scores['neg_score']:
        features['pos_ratio'] += 1
      elif scores['neg_score'] > scores['pos_score']:
        features['neg_ratio'] += 1
  features['pos_ratio'] = features['pos_ratio'] / w_count if w_count else 0 
  features['neg_ratio'] = features['neg_ratio'] / w_count if w_count else 0
    
  return features


def group_reviews_by_product(reviews):
  """ Groups review by product.

      Args:
        reviews: a dictionary of reviews.

      Returns:
        A dictionary indexed firstly by product name and secondly by review ids,
      containing review dictionaries as second order values.
  """
  grouped_reviews = {}

  for review in reviews.values():
    if review['product'] not in grouped_reviews:
      grouped_reviews[review['product']] = []
    grouped_reviews[review['product']].append(review)

  return grouped_reviews


def calculate_kl_divergence(train_reviews, reviews):
  """ Calculates KL divergence between unigram models of review's text and all
      reviews' texts from the corresponding product.

      Args:
        train_reviews: a set of reviews which are in training set.
        reviews: a dictionary of reviews.

      Returns:
        None. The KL divergence value is create in each review dictionary under
      the key 'kl'.
  """
  grouped_reviews = group_reviews_by_product(reviews)
  for product in grouped_reviews:
    avg_unigram = {}
    avg_text = ""
    for review in grouped_reviews[product]:
      review['unigram'] = get_unigram_model(review['text'])
      if review['id'] in train_reviews:
        avg_text += " " + review['text'] # average only relative to train
    avg_unigram = get_unigram_model(avg_text)
    for review in grouped_reviews[product]:
      if review['id'] in train_reviews:
        review['kl'] = 0
        for word in review['unigram']:
          review['kl'] += review['unigram'][word] * log(review['unigram'][word] /
              avg_unigram[word])
        review.pop('unigram', None)
      else:
        # if review is in test, than it is like there is just this review
        # and the others from training set 
        review['kl'] = 0
        new_avg_unigram = add_unigrams(avg_unigram, review['unigram'])
        for word in review['unigram']:
          review['kl'] += review['unigram'][word] * log(review['unigram'][word] /
              new_avg_unigram[word])
        review.pop('unigram', None)


def add_unigrams(unigram_a, unigram_b):
  """ Adds two unigram models into a unified one.

      Args:
        unigram_a: dictionary representing a unigram, indexed by a word (string)
      and containing float, the relative frequency, as values.
        unigram_b: dictionary representing a unigram, indexed by a word (string)
      and containing float, the relative frequency, as values.

      Returns:
        A dictionary representing a unigram, indexed by a string, the word, and
      having a float with the relative frequency as value.
  """
  new_unigram = unigram_a.copy()
  n_a = len(unigram_a)
  n_b = len(unigram_b)
  words = set(unigram_a.keys()).union(set(unigram_b.keys()))
  n_new = len(words)
  for word in words:
    if word in unigram_a and word in unigram_b: 
      new_unigram[word] = (unigram_a[word] * n_a + unigram_b[word] * n_b) / \
          n_new
    elif word in unigram_a:
      new_unigram[word] = (unigram_a[word] * n_a) / n_new
    else:
      new_unigram[word] = (unigram_b[word] * n_b) / n_new
  return new_unigram


def calculate_kl_divergence_db(train_reviews):  
  """ Calculates KL divergence between unigram models of review's text and all
      reviews' texts from the corresponding product. Using data from database.

      Args:
        train_reviews: a set of reviews which are in training set.

      Returns:
        None. The KL divergence value is create in each review dictionary under
      the key 'kl'.
  """
  distinct_products = reviews_db.distinct("product")
  for product in distinct_products:
    avg_unigram = {}
    avg_text = ""
    for review in reviews_db.find({"product": product}):
      review['unigram'] = get_unigram_model(review['text'])
      reviews_db.update({'_id': review["_id"]} , {"$set": review}, upsert=True)
      if review['id'] in train_reviews:
        avg_text += " " + review['text'] # average only relative to train
    avg_unigram = get_unigram_model(avg_text)
    for review in reviews_db.find({"product": product}):
      review['kl'] = 0
      for word in review['unigram']:
        review['kl'] += review['unigram'][word] * log(review['unigram'][word] /
            avg_unigram[word])
      reviews_db.update({'_id': review["_id"]} , {"$set": review}, upsert=True)


def get_unigram_model(text):
  """ Gets an unigram model for a given text.
      
     Args:
        text: a string with the text to get the model of.

      Returns:
        A dictionary with the frequencies :of words.
  """
  unigram = {}
  text =  text.replace(".", " ")
  text_blob = TextBlob(text)

  for word in text_blob.words:
    if word.decode() not in unigram:
      unigram[word.decode()] = 0.0
    unigram[word.decode()] += 1.0

  for word in unigram:
    unigram[word] = unigram[word] / len(text_blob.words) if \
        len(text_blob.words) else 0

  return unigram


def model_products(train_reviews, reviews):
  """ Extracts and models products from a set of reviews.

      Args:
        train_reviews: a set with reviews' ids which are in training set.
        reviews: a list of review dictionaries.

      Returns:
        A dictionary of products indexed by product name and containing a
      dictionary with key 'avg_rating' as value.
  """
  products = {}
  for review in reviews.values():
    product = review['product']
    if product not in products:
      products[product] = {}
      products[product]['sum'] = products[product]['count'] = 0.0
    if review['id'] in train_reviews: # average regarding only train
      products[product]['sum'] += review['rating']
      products[product]['count'] += 1.0
  for product in products:
    products[product]['avg_rating'] = products[product]['sum'] / \
        products[product]['count'] if products[product]['count'] > 0 else nan
  return products


def calculate_rel_rating(train_reviews, reviews):
  """ Calculates relative rating of reviews, which consists in the rating minus
      the product average rating.

      Args:
        reviews: dictionary of reviews.

      Returns:
        None. Dictionary of reviews is changed in place with an additional key
      'rel_rating' per review .
  """
  products = model_products(train_reviews, reviews)
  for r_id in reviews:
    review = reviews[r_id]
    p_id = review['product']
    product = products[p_id]
    if r_id in train_reviews:
      review['rel_rating'] = review['rating'] - product['avg_rating']
    else:
      # In test, the average rating includes the train ratings and an example
      if isnan(product['avg_rating']):
        avg_rating = review['rating']
      else:
        avg_rating = (product['sum'] + review['rating']) / \
            (product['count'] + 1)
      review['rel_rating'] = review['rating'] - avg_rating 


def calculate_avg_vote(train_reviews, reviews):
  """ Calculates average vote of the helpfulness votes received by each review. 

      Args:
        train_reviews: set of reviews' ids which are in train.
        reviews: dictionary of reviews.

      Returns:
        None. Dictionary of reviews is changed in place with an additional key
      'rel_rating' per review .
  """
  for r_id in reviews:
    reviews[r_id]['avg_vote'] = mean(reviews[r_id]['votes'].values()) if \
        reviews[r_id]['votes'] else nan

def calculate_rel_rating_db(train_reviews, reviews):
  """ Calculates relative rating of reviews, which consists in the rating minus
      the product average rating. Using database.

      Args:
        reviews: dictionary of reviews.

      Returns:
        None. Dictionary of reviews is changed in place with an additional key
      'rel_rating' per review .
  """
  products = model_products(train_reviews, reviews)
  for review_id in reviews:
    product = reviews[review_id]['product']
    reviews[review_id]['rel_rating'] = reviews[review_id]['rating'] - \
        products[product]['avg_rating']


def model_reviews_parallel(num_threads, train, reviews):
  """ Models reviews in parallel using num_threads threads.

      Args:
        num_threads: number of parallel jobs.
        train: a list of votes in training set.
        reviews: dictionary with parsed raw reviews to add features in. 

      Returns:
        None. Changes are made in place in reviews dictionary. 
  """
  texts = [(r_id, reviews[r_id]['text']) for r_id in reviews]

  pool = Pool(processes=num_threads)
  chunk_size = int(ceil(len(texts) / num_threads))
  result = pool.imap_unordered(model_review_text, iter(texts), chunk_size)
  pool.close()
  pool.join()
  result = [r for r in result]
  train_reviews = set([vote['review'] for vote in train])
  for r_id, r_dict in result:
    for feat in r_dict:
      reviews[r_id][feat] = r_dict[feat]
  if _USE_DB:
    calculate_kl_divergence_db(train_reviews)
    calculate_rel_rating_db(train_reviews)
  else:
    calculate_kl_divergence(train_reviews, reviews)  
    calculate_rel_rating(train_reviews, reviews)
  calculate_avg_vote(train_reviews, reviews)
