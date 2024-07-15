import sys
import os
import numpy as np
import pandas as pd
import psycopg2
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

df = pd.read_csv('./pharmacogenomic_drug_targets.tsv.gz', sep='\t')
drug_targets = set([])
for r in df.values:

    ups = set([])
    if not pd.isnull(r[9]):
        ups.update(r[9].split('|'))
    if not pd.isnull(r[11]):
        ups.update(r[11].split('|'))
    if not pd.isnull(r[13]):
        ups.update(r[13].split('|'))

    if len(ups) == 0: continue
    for up in ups:
        drug_targets.add((r[0], up))

#Writing
with open(out_path+'/CPD-int-GEN/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for dg, g in sorted(drug_targets):
        o.write('%s\t%s\n'%(dg,g))
        
sys.stderr.write('Done!\n')
