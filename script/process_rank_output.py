from math import sqrt
from sys import argv

infile = open(argv[1], "r")
outfile = open(argv[2], "r")

sse = 0.0
norm_sse = 0.0
count = 0
for inline in infile:
  outline = outfile.readline()
  truth = float(int(inline.strip().split()[0]))
  pred = float(outline.strip())
  sse += (truth - pred) ** 2  
  norm_sse += (truth / 5.0 - pred / 5.0) ** 2
  count += 1
infile.close()
outfile.close()

rmse = sqrt(sse / count)
norm_rmse = sqrt(norm_sse / count)

print "RMSE: %f" % rmse
print "RMSE: %f (normalized)" % norm_rmse
