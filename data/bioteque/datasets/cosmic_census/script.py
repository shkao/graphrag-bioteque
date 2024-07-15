import sys
import os
import numpy as np
import pandas as pd
import gzip
import networkx as nx
from tqdm import tqdm
from scipy.stats.mstats import gmean
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#--------Parameters-----------

tiers = set([1,2])

#------------------------------

#--Rading
m = pd.read_csv('./CosmicMutantExportCensus.tsv.gz', sep='\t', encoding= 'unicode_escape')

#--Filtering
m = m[(m['Sample Type'] == 'cell-line') & (m['Tier'].isin(tiers))]
m = m[['Sample name','HGNC ID']].values

#--Mapping
cls = sorted(set(m[:,0]))
cl2id = dict(zip(cls,mps.cl2ID(cls)))
g2id = mps.get_geneID2unip()

#--Getting edges
edges = set([])
for r in tqdm(m, desc='Processing'):
    cl = cl2id[r[0]]
    g = str(r[1])
    
    if g not in g2id: continue
    if cl == None: continue
    
    for up in g2id[g]:
        edges.add((cl,up))
    
#--Removing disconnected mutations (if any)
G = nx.Graph()
G.add_edges_from(edges)
uv = sorted(nx.connected_components(G), key = lambda x: len(x), reverse=True)[0]

#--Writing
with open(out_path+'/CLL-mut-GEN/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for cl,g in sorted(edges):
        if cl not in uv or g not in uv: continue

        o.write('%s\t%s\n'%(cl,g))
sys.stderr.write('Done!\n')

