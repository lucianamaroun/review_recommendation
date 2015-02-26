""" Review modeling module
    ----------------------

    Models reviews with a set of features derived from rating and text.

    Usage:
      Used only as a module, not directly callable.
"""


from string import punctuation
from math import log
from copy import deepcopy
from multiprocessing import Pool
from re import match

from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk import pos_tag
from nltk.corpus import wordnet
import traceback

from src import parser
from src.lib.sentiment.sentiwordnet import SimplifiedSentiWordNet


_NEGATION = set(['not', 'no', 'n\'t'])
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


""" Models a review.

    Args:
      raw_review: A dictionary representing a review.

    Returns:
      A dictionary with more keys represented new modeled attributes.
"""
def model_review(raw_review):
  try:
    review = deepcopy(raw_review)
    if get_foreign_ratio(review['text']) >= 0.4:
      return None
    text_feat = get_textual_features(review['text'])
    for feat in text_feat:
        review[feat] = text_feat[feat]
    return review
  except Exception as e:
    traceback.print_exc()
    print()
    raise e


""" Get ratio of foreign (or unidentified) words in a text.

    Args:
      text: a string containing the text.

    Returns:
      A real value with the ratio of foreign words.
"""
def get_foreign_ratio(text):
  text = text.lower()
  word_tokenizer = RegexpTokenizer(r'\w+')
  words = word_tokenizer.tokenize(text)
  fw = [w for w in words if match('[a-z]+', w) and wordnet.synsets(w) == []]
  if len(words) > 0:  
    return float(len(fw)) / len(words)
  else:
    return 0


""" Reports different features derived from text, related to length, syntax,
    lexicon and sentiment statistics.

    Args:
        text: the text to derive the features from.

    Returns:
        A dictionary from feature names to feature value. Refer to
    "get_text_length_stats", "get_pos_stats" and "get_sent_stats" for complete
    list of feature names.
"""
def get_textual_features(text):
    for sub, target in _SUBSTITUTE.items():
      text = text.replace(sub, target)
    lower_text = text.lower()

    features = {}
    length_feat = get_text_length_stats(text)
    for feat in length_feat:
        features[feat] = length_feat[feat]

    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    pos_feat = get_pos_stats(tags)
    for feat in pos_feat:
        features[feat] = pos_feat[feat]

    sent_feat = get_sent_stats(tags)
    for feat in sent_feat:
      features[feat] = sent_feat[feat]

    return features


""" Gets length statistics of the text: number of chars, number of tokens,
    number of words, number of sentences, ratio of unique words, average number
    of tokens by sentence and ratio of capitalized sentences.

    Args:
        text: the text to calculate statistics from.

    Returns:
        A dictionary indexed by keys "num_chars", "num_tokens", "num_words",
    "num_sents", "uni_ratio", "avg_sent" and "cap_sent", and containing as
    values the corresponding statistic value.
"""
def get_text_length_stats(text):
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
def get_pos_stats(tags):
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


""" Gets positive and negative ratio of sentences and words. Sentence ratio
    has a better context for deriving meaning and, consequently, sentiment. Word
    related ratio is calculated using the first meaning of the identified tag.

    Args:
      tags: list of pairs (word, tag). 

    Returns:
      Two real values, pos_ratio and neg_ratio, with, respectively the ratio of
    positive sentences and the ratio of negative sentences.
"""
def get_sent_stats(tags):
  features = {}
  negate = False

  swn = SimplifiedSentiWordNet()
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
    scores = swn.scores(word, tag)
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


""" Groups review by product.

    Args:
      reviews: a dictionary of reviews.

    Returns:
      A dictionary indexed firstly by product name and secondly by review ids,
    containing review dictionaries as second order values.
"""
def group_reviews_by_product(reviews):
  grouped_reviews = {}

  for review in reviews.values():
    if review['product'] not in grouped_reviews:
      grouped_reviews[review['product']] = []
    grouped_reviews[review['product']].append(review)

  return grouped_reviews


""" Calculates KL divergence between unigram models of review's text and all
    reviews' texts from the corresponding product.

    Args:
      reviews: a dictionary of reviews.

    Returns:
      None. The KL divergence value is create in each review dictionary under
    the key 'kl'.
"""
def calculate_kl_divergence(reviews):
  grouped_reviews = group_reviews_by_product(reviews)

  avg_unigram = {}
  total_words = 0
  for product in grouped_reviews:
    for review in grouped_reviews[product]:
      review['unigram'], num_words = get_unigram_model(review['text'])
      for word in review['unigram']:
        if word not in avg_unigram:
          avg_unigram[word] = 0
        avg_unigram[word] += review['unigram'][word]
      total_words += num_words
    for word in avg_unigram:
      avg_unigram[word] = avg_unigram[word] / total_words if total_words else 0
    for review in grouped_reviews[product]:
      review['kl'] = 0
      for word in review['unigram']:
        if not avg_unigram[word]:
          continue
        review['kl'] += review['unigram'][word] * log(review['unigram'][word] /
            avg_unigram[word])
      review.pop('unigram', None)


""" Gets an unigram model for a given text.

    Args:
      text: the text to get the model of.

    Returns:
      A dictionary with the counts of words and the number of words.

    Obs.: The unigram model is not retrieved ready, with counts normalized to
    ratios, because the intermediate representations makes it easier to
    aggregate unigram models for all reviews of a product.
"""
def get_unigram_model(text):
  unigram = {}
  text_blob = TextBlob(text)

  for word in text_blob.words:
    if word.decode() not in unigram:
      unigram[word.decode()] = 0.0
    unigram[word.decode()] += 1.0

  for word in unigram:
    unigram[word] = unigram[word] / len(text_blob.words) if \
        len(text_blob.words) else 0

  return unigram, len(text_blob.words)


""" Extracts and models products from a set of reviews.

    Args:
      reviews: A list of review dictionaries.

    Returns:
      A dictionary of products indexed by product name and containing a
    dictionary with key 'avg_rating' as value.
"""
def model_products(reviews):
  products = {}
  for review in reviews.values():
    product = review['product']
    if product not in products:
      products[product] = {}
      products[product]['sum'] = products[product]['count'] = 0.0
    products[product]['sum'] += review['rating']
    products[product]['count'] += 1.0
  for product in products:
    products[product]['avg_rating'] = products[product]['sum'] / \
        products[product]['count']
  return products


""" Calculates relative rating of reviews, which consists in the rating minus
    the product average rating.

    Args:
      reviews: dictionary of reviews.

    Returns:
      None. Dictionary of reviews is changed in place with an additional key
    'rel_rating' per review .
"""
def calculate_rel_rating(reviews):
  products = model_products(reviews)
  for review_id in reviews:
    product = reviews[review_id]['product']
    reviews[review_id]['rel_rating'] = reviews[review_id]['rating'] - \
        products[product]['avg_rating']


""" Models reviews in parallel using num_threads threads.

    Args:
      num_threads: number of parallel jobs.
      sample_raw_reviews: (optional) The sample set of reviews to be modeled.

    Returns:
      A dictionary of reviews indexed by reviews' ids.
"""
def model_reviews_parallel(num_threads, sample_raw_reviews=None):
  raw_reviews = sample_raw_reviews if sample_raw_reviews else [r for r in
      parser.parse_reviews()]

  pool = Pool(processes=num_threads)
  result = pool.imap_unordered(model_review, iter(raw_reviews), len(raw_reviews)
      / num_threads + 1)
  pool.close()
  pool.join()
  result = [review for review in result if review]
  reviews = {review['id']:review for review in result}

  calculate_kl_divergence(reviews)
  calculate_rel_rating(reviews)

  return reviews

