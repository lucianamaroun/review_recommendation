from preprocessing.parsing import parse_reviews, parse_trusts

trusts = parse_trusts()
count_authors = 0
count_not_trust = 0
for review in parse_reviews():
  if review['author'] not in trusts:
    count_not_trust += 1
  count_authors += 1

print '# of authors: %d' % count_authors
print '# of users in trust: %d' % trusts.number_of_nodes()
print '# of authors not in trust: %d' % count_not_trust
