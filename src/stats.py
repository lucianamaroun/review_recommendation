""" Stats script
    ------------

    Obtains basic statiscts of the dataset.

    Usage:
      $ python -m src.stats
    on the root directory of the project.
"""

from src import parser

num_rev = 0
num_voted = 0
num_voted5 = 0
num_voted10 = 0
total_votes = 0
users = set()
products = set()
for r in parser.parse():
  num_rev += 1
  user = r['user']
  if user not in users:
    users.add(user)
  product = r['product']
  if product not in products:
    products.add(product)
  if r['num_votes'] > 0:
    num_voted += 1
  if r['num_votes'] >= 5:
    num_voted5 += 1
  if r['num_votes'] >= 10:
    num_voted10 += 1
  total_votes += r['num_votes']
avg_votes = float(total_votes) / float(num_rev)
min_date = max_date = None
for review in parser.iterate_reviews():
  date = datetime.datetime.strptime(review['date'], '%d.%m.%Y')
  if not min_date or (min_date and min_date > date):
    min_date = date
  if not max_date or (max_date and max_date < date):
    max_date = date

print '# reviews: %d' % num_rev
print '# users: %d' % len(users)
print '# products: %d' % len(products)
print '# helpfulness votes: %d' % total_votes
print '# voted reviews: %d' % num_voted
print '# voted by 5 or more reviews: %d' % num_voted5
print '# voted by 10 or more reviews: %d' % num_voted10
print '# helpfulness votes by review: %d' % avg_votes
print 'reviews\' timespan %s - %s' % (min_date.strftime('%d.%m.%Y'),
    max_date.strftime('%d.%m.%Y'))
