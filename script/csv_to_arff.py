""" Csv to Arff Script
    ------------------

    Transforms a csv set of instances in a arff set of instances. For
    different files, change _INS and _OUTS variables. 
    
    Usage:
      $ python -m script.csv_to_arff
    on the root directory of the project.
"""

import csv


_INS = ['/var/tmp/luciana/train1.csv', '/var/tmp/luciana/test1.csv']
_OUTS = ['/var/tmp/luciana/train1.arff', '/var/tmp/luciana/test1.arff']

for i in [0, 1]:
  with open(_INS[i], 'r') as data:
    out = open(_OUTS[i], 'w')
    print >> out, \
'''@relation train_v1

@attribute rating {0, 1, 2, 3, 4, 5}
@attribute num_chars numeric
@attribute num_tokens numeric
@attribute num_words numeric
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
@attribute pos_ratio numeric
@attribute neg_ratio numeric
@attribute kl_div numeric
@attribute r_num_reviews numeric
@attribute r_avg_rating numeric
@attribute r_avg_help_rec numeric
@attribute r_num_trustors numeric
@attribute r_num_trustees numeric
@attribute r_avg_help_giv numeric
@attribute r_avg_rel_help_giv numeric
@attribute r_sd_rating numeric
@attribute r_sd_help_rec numeric
@attribute r_sd_help_giv numeric
@attribute r_pagerank numeric
@attribute u_num_reviews numeric
@attribute u_avg_rating numeric
@attribute u_avg_help_rec numeric
@attribute u_num_trustors numeric
@attribute u_num_trustees numeric
@attribute u_avg_help_giv numeric
@attribute u_avg_rel_help_giv numeric
@attribute u_sd_rating numeric
@attribute u_sd_help_rec numeric
@attribute u_sd_help_giv numeric
@attribute u_pagerank numeric
@attribute trust {0, 1}
@attribute class {0, 1, 2, 3, 4, 5}

@data'''
    reader = csv.reader(data)
    reader.next()
    for l in reader:
      print >> out, ','.join(['%s'] * 46) % tuple(l[3:])
