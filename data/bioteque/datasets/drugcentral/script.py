#DRUGCENTRAL TARGETS
import os
import sys
import pandas as pd
import numpy as np
import subprocess
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps
from transform_data import find_outliers_from_edges
gene_uv = mps.get_human_reviewed_uniprot()

current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]
out_path = '../../graph/raw/CPD-int-GEN/'
mapping_path = '../../metadata/mappings/'

#--Download the data
subprocess.Popen('./get_data.sh', shell = True).wait()

#--Mapping to inchikey
d2ikey = {}
with open(mapping_path+'/CPD/drugcentral.tsv', 'r') as f:
    for l in f:
        l = l.rstrip("\n").split("\t")
        if not l[1]: continue
        d2ikey[l[0]] = l[1]


#--Readding drug targets
df = pd.read_csv('./drug.target.interaction.tsv', sep='\t')
df = df[df['ORGANISM']=='Homo sapiens']

#Mapping edges and keeping drug-target interactions
edges = set([])
for r in df.values:
    r[1] = str(r[1])
    if r[1] not in d2ikey: continue
    ikey = d2ikey[r[1]]
    gn = r[4]
    if gn not in gene_uv: continue
    edges.add((ikey,gn))

#--Finding outliers and removing them
n1_outliers, n2_outliers = find_outliers_from_edges(edges, False)
edges = set([(n1,n2) for n1,n2 in edges if n1 not in n1_outliers and n2 not in n2_outliers])

#--Writing
with open(out_path+'/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for a,b in sorted(edges):
        o.write('%s\t%s\n'%(a,b))

sys.stderr.write('Done!\n')
