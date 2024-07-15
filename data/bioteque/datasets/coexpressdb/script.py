import os
import sys
import pandas as pd
import numpy as np
import subprocess
from scipy.stats.mstats import gmean
from tqdm import tqdm
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps
out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]
p = './Hsa-r.v18-12.G22897-S22897.combat_pca_subagging.mrgeo.d/'

#--Download the data
subprocess.Popen('./get_data.sh', shell = True).wait()

#---------- Parameters ----------

cutoff = np.ceil(len(os.listdir(p)))*0.1 # ~ 10% of total possible genes

#--------------------------------


#--Getting reliable genes
reliable_genes = set([])
with open('./supportability.2014-08-19.txt','r') as f:
    for l in f:
        if not l.endswith('Hsa3\n'):continue
        h = l.rstrip().split('\t')
        if int(h[1]) == 0: continue
        reliable_genes.add(h[0])

#--mapping
genes = os.listdir(p)
gn2up = mps.get_geneID2unip()
gn2up = {x:gn2up[x] for x in genes if x in gn2up}
pairs = {}

for source_gene in tqdm(os.listdir(p), desc='Genes'):
    if source_gene not in gn2up: continue
    with open(p+'/%s'%source_gene,'r') as f:
        for l in f:
            h = l.rstrip().split('\t')
            if h[0] not in gn2up or h[0] not in reliable_genes: continue
            if float(h[1]) < cutoff:
                for g1 in gn2up[source_gene]:
                    for g2 in gn2up[h[0]]:
                        if g1 == g2: continue
                        cex = tuple(sorted([g1,g2]))
                        score = float(h[1])
                        if cex in pairs:
                            pairs[cex] = gmean([pairs[cex],score])
                        else:
                            pairs[cex] = score

with open(out_path+'/GEN-cex-GEN/%s.tsv'%(source),'w') as o:
    o.write('n1\tn2\tmutual_rank\n')
    for pair in sorted(pairs):
         o.write('%s\t%s\t%i\n'%(pair[0],pair[1],pairs[pair]))
