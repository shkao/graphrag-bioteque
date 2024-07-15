#REPORPUSING HUB TARGETS (LINCS)
import os
import sys
import pandas as pd
import numpy as np
import subprocess
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps
from transform_data import find_outliers_from_edges
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]
out_path = '../../graph/raw/CPD-int-GEN/'
mapping_path = '../../metadata/mappings/'

g2g = mps.get_gene2updatedgene()
g2u = mps.get_gene2unip()

#--Download the data
subprocess.Popen('./get_data.sh', shell = True).wait()

#--Mapping to inchikey
d2ikey = {}
with open(mapping_path+'/CPD/repohub.tsv', 'r') as f:
    for l in f:
        l = l.rstrip("\n").split("\t")
        if not l[1]: continue
        d2ikey[l[0]] = l[1]


#--Reading drugtarget data
with open('./repurposing_drugs_20200324.txt', 'r') as f:
    m = []
    flag = False
    for l in f:

        if flag:
            m.append(l.rstrip('\n').split('\t'))

        if flag is False and not l.startswith('!') and l != '':
            flag = True
            hd = l.rstrip('\n').split('\t')
    df = pd.DataFrame(m, columns=hd)

#--Getting edges
edges = set([])
k = set([])
id_ix = np.where(df.columns == 'pert_iname')[0][0]
tg_ix = np.where(df.columns == 'target')[0][0]
for r in df.values:
    if not r[id_ix] in d2ikey:continue

    ikey = d2ikey[r[id_ix]]
    k.add(ikey)
    for g in r[tg_ix].split('|'):
        if g in g2g:
            g = g2g[g]
        if g in g2u:
            for up in g2u[g]:
                edges.add((ikey, up))

#--Finding outliers and removing them
n1_outliers, n2_outliers = find_outliers_from_edges(edges, False)
edges = set([(n1,n2) for n1,n2 in edges if n1 not in n1_outliers and n2 not in n2_outliers ])

#--Writing
with open(out_path+'/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for a,b in sorted(edges):
        o.write('%s\t%s\n'%(a,b))

sys.stderr.write('Done!\n')
