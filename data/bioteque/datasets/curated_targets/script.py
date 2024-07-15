import sys
import os
sys.path.insert(0, '../../code/kgraph/utils/')
from transform_data import find_outliers_from_edges

out_path = '../../graph/raw/CPD-int-GEN/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#-------Parameters-------
curated_datasets = ['drugbank_pd', 'pharmacogenomic_targets', 'repohub']
zscore_cutoff = 10 # --> To remove extreme outliers
#------------------------


edges = set([])
for dt in curated_datasets:
    dt = dt+'.tsv'
    if not os.path.exists(out_path+dt):
        sys.stderr.write('Dataset %s not found in path. Skipping...\n'%dt)
        continue
    with open(out_path+dt,'r') as f:
        next(f)
        for l in f:
            edges.add(tuple(l.rstrip().split('\t')))

n1_outliers, n2_outliers = find_outliers_from_edges(edges, False, zscore_cutoff=zscore_cutoff)
edges = set([(n1,n2) for n1,n2 in edges if n1 not in n1_outliers and n2 not in n2_outliers])

with open(out_path+'/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for dg,g in sorted(edges):
        o.write('%s\t%s\n'%(dg,g))
