import os
import sys
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait() # <-- This may fail! You can manually download the data from: http://ctdbase.org/downloads/;jsessionid=FA0CD81392EA4E24D78F2E6C56B2F76D#gd 

#Reading the data
m = set([])
hd = ['GeneSymbol','GeneID','DiseaseName','DiseaseID','DirectEvidence','InferenceChemicalName','InferenceScore']
with open('./CTD_genes_diseases.tsv', 'r') as f:
    for _ in range(29):
        next(f)
    for l in tqdm(f, desc='Reading file'):
        d = dict(zip(hd,l.rstrip('\n').split('\t')))
        
        if d['DirectEvidence'] == '': continue
        m.add((d['GeneID'], d['DiseaseID']))
m = np.asarray(list(m))

#Mapping geneID
genes = list(map(str,m[:,0]))
gn2up = mps.get_geneID2unip()
gn2up = {x:gn2up[x] for x in genes if x in gn2up}

#Removing genes that were not mapped
m = m[[ix for ix,x in enumerate(m) if x[0] in gn2up]]

#Iterating through the data and writing
pairs = set([])
for r in m:
    for g in gn2up[r[0]]:
        pairs.add((g,r[1]))
        
#Writing
with open(out_path+'/GEN-ass-DIS/%s.tsv'%(source),'w') as o:
    o.write('n1\tn2\n')
    for r in pairs:
        o.write('\t'.join(r)+'\n')

sys.stderr.write('Done!\n')
