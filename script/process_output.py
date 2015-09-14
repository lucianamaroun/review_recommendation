from math import sqrt
from sys import argv, exit
from pickle import load

from src.util.evaluation import calculate_ndcg

if len(argv) != 4:
  print ('Usage: python -m script.process_output <test_pickle_file> '
      '<prediction_text_file> <output_text_file>')
  exit()
test = load(open(argv[1], 'r'))
predfile = open(argv[2], 'r')

pred = []
sse = 0.0
norm_sse = 0.0
count = 0
for vote in test:
  predline = predfile.readline()
  truth = float(vote['vote'])
  prediction = float(predline.strip())
  pred.append(prediction)
  sse += (truth - prediction) ** 2  
  norm_sse += (truth / 5.0 - prediction / 5.0) ** 2
  count += 1
predfile.close()

rmse = sqrt(sse / count)
norm_rmse = sqrt(norm_sse / count)

output = open(argv[3], 'w')
print "RMSE: %f" % rmse
print "RMSE: %f (normalized)" % norm_rmse
print >> output, "RMSE: %f" % rmse
print >> output, "RMSE: %f (normalized)" % norm_rmse

pred_group = {}
truth_group = {}
for i in xrange(len(test)):
  voter = test[i]['voter']
  if voter not in pred_group:
    pred_group[voter] = []
    truth_group[voter] =[]
  pred_group[voter].append(pred[i])
  truth_group[voter].append(test[i]['vote'])
for i in xrange(1, 21):
  score_sum = 0.0
  sizes = []
  for key in pred_group:
    score_sum += calculate_ndcg(pred_group[key], truth_group[key], i)
    sizes.append(len(pred_group[key]))
  score = score_sum / len(pred_group)
  print 'NDCG@%d: %f (grouped V)' % (i, score)
  print >> output, 'NDCG@%d: %f (grouped V)' % (i, score)
sizes = []
for key in pred_group:
  sizes.append(len(pred_group[key]))
hist = {s: float(sizes.count(s)) / len(sizes) for s in sizes}
print >> output, "Histogram of ranking sizes (grouped V): ",
print >> output, hist

for i in xrange(1, 21):
  score = calculate_ndcg(pred, [v['vote'] for v in test], i)
  print 'NDCG@%d: %f (all)' % (i, score)
  print >> output, 'NDCG@%d: %f (all)' % (i, score)

output.close()
