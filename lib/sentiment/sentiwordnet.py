""" SentiWordNet Module
    -------------------
    
    Contains a simplified sentiwordnet wrapper which uses the first
  meaning of a word under a given tag.
  
    This module is used by other modules and should not be directly called.
"""

FILE = 'lib/sentiment/SentiWordNet_3.0.0.txt' # default

class SimplifiedSentiWordNet(object):
  """ Simplified SentiWordNet wrapper. It is simplified because it considers the first meaning for a word under a given tag.
  
      Args:
        self: the SimplifiedSentiWordNet wrapper object.
        input_file: The file containing SentiWordNet data.
  """
  def __init__(self, input_file=FILE):
    self._parse(input_file)

  """ Parses the data file, creating a readily available dictionary
      indexed firstly by word, secondly by tag and lastly by 'pos'
      and 'neg', having polarity float scores as highest order values.
      
      Args:
        self: the SimplifiedSentiWordNet wrapper object.
        input_file: The file containing SentiWordNet data.
      
      Returns:
        None.
  """
  def _parse(self, input_file):
    self.sentiment = {}
    with open(input_file) as swnfile:
      line = swnfile.readline()
      while (line[0] == '#'):
        line = swnfile.readline()
      while line:
        if line[0] == '\t':
          line = swnfile.readline()
          continue
        tag, _, pos, neg, words_raw, _ = line.split('\t')
        try:
          pos, neg = float(pos), float(neg)
        except:
          print line
        words = [w.split('#')[0] for w in words_raw.split(' ') if w.split('#')[1] == '1']
        for word in words:
          if word not in self.sentiment:
            self.sentiment[word] = {}
          self.sentiment[word][tag] = {'pos_score': pos, 'neg_score': neg}
        line = swnfile.readline()
  
  """ Looks up the positive and negative scores of a word classified as certain tag.
  
      Args:
        word: the string with the word whose polarity is desirable.
        tag: the classified tag of the word.
      
      Returns:
        A dictionary with keys 'pos' and 'neg' containing the positive and negative scores of the word. If the word is not in SentiWordNet, then None is returned.
  """
  def scores(self, word, tag):
    return self.sentiment[word][tag] if word in self.sentiment and \
        tag in self.sentiment[word] else None
