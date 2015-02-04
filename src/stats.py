""" Stats script
    ------------

    Obtains basic statiscts of the dataset.

    Usage:
      $ python -m src.stats
    on the root directory of the project.
"""

from src import parser
from datetime import datetime


num_rev = 0
num_voted = 0
num_voted5 = 0
num_voted10 = 0
total_votes = 0
missing_prod = 0
reviewers = set()
raters = set()
products = set()
min_date = max_date = None
for r in parser.parse_reviews():
  num_rev += 1
  rvr = r['user']
  if rvr not in reviewers:
    reviewers.add(rvr)
  for rtr in r['votes']:
    if rtr not in raters:
      raters.add(rtr)
  product = r['product']
  if not product:
    missing_prod += 1
  elif product not in products:
    products.add(product)
  num_votes = len(r['votes'])
  if num_votes > 0:
    num_voted += 1
  if num_votes >= 5:
    num_voted5 += 1
  if num_votes >= 10:
    num_voted10 += 1
  total_votes += num_votes
  date = datetime.strptime(r['date'], '%d.%m.%Y')
  if not min_date or (min_date and min_date > date):
    min_date = date
  if not max_date or (max_date and max_date < date):
    max_date = date
avg_votes = float(total_votes) / float(num_rev)

print '# reviews: %d' % num_rev
print '# users: %d' % len(reviewers.union(raters))
print '# reviewers: %d' % len(reviewers)
print '# raters: %d' % len(raters)
print '# reviewers & raters: %d' % len(reviewers.intersection(raters))
print '# products: %d' % len(products)
print '# reviews w/ missing products: %d' % missing_prod
print '# helpfulness votes: %d' % total_votes
print '# voted reviews: %d' % num_voted
print '# voted by 5 or more reviews: %d' % num_voted5
print '# voted by 10 or more reviews: %d' % num_voted10
print 'avg # votes by review: %d' % avg_votes
print 'reviews\' timespan %s - %s' % (min_date.strftime('%d.%m.%Y'),
    max_date.strftime('%d.%m.%Y'))
