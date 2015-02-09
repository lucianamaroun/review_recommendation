import csv


ins = ['/var/tmp/luciana/train-notustat.txt', '/var/tmp/luciana/test-notustat.txt']
outs = ['/var/tmp/luciana/train.arff', '/var/tmp/luciana/test.arff']

for i in [0, 1]:
  with open(ins[i], 'r') as data:
    out = open(outs[i], 'w')
    print >> out, \
'''@relation train_v1

@attribute num_tokens numeric
@attribute num_sents numeric
@attribute unique_ratio numeric
@attribute avg_sent numeric
@attribute cap_ratio numeric
@attribute noun_ratio numeric
@attribute adj_ratio numeric
@attribute adv_ratio numeric
@attribute verb_ratio numeric
@attribute comp_ratio numeric
@attribute fw_ratio numeric
@attribute sym_ratio numeric
@attribute num_ratio numeric
@attribute punct_ratio numeric
@attribute pos_sent numeric
@attribute neg_sent numeric
@attribute r_num_reviews numeric
@attribute r_avg_rating numeric
@attribute r_avg_rel_rating numeric
@attribute r_avg_help_rec numeric
@attribute r_num_trustors numeric
@attribute r_num_trustees numeric
@attribute u_avg_rating numeric
@attribute u_avg_rel_rating numeric
@attribute u_num_trustors numeric
@attribute u_num_trustees numeric
@attribute u_avg_help_giv numeric
@attribute u_avg_rel_help_giv numeric
@attribute trust {0, 1}
@attribute class {0, 1, 2, 3, 4, 5}

@data'''
    reader = csv.reader(data)
    reader.next()
    for l in reader:
      print >> out, \
      '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s' % tuple(l[3:])
