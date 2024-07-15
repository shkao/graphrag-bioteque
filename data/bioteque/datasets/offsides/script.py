import os
import sys
import subprocess
import pandas as pd
import numpy as np
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#------------Parameters----------

PRR_cutoff = 2
PRR_error = 0.25

#--------------------------------

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait()

#Reading the data
df = pd.read_csv('./OFFSIDES.csv')

#--Removing unnanotated PRR
for lb in ['PRR', 'PRR_error']:
    v = []
    for x in df[lb]:
        try:
            v.append(float(x))
        except:
            v.append(np.nan)
    df[lb] = np.array(v, dtype=float)
    
del v

df = df[(df['PRR']>=PRR_cutoff) & (df['PRR_error']<=PRR_error)]

#Mapping drugs
db2ikey = mps.get_drugbank2ikey()
k = pd.read_csv('../../metadata/mappings/CPD/rxnorm.tsv', sep='\t')
_d = dict(zip(k['rxnorm_id'], k['drugbank_id']))
d2ik = {x:db2ikey[_d[x]] for x in _d if _d[x] in db2ikey}
del _d,k
df['cpd'] = [d2ik[x] if x in d2ik else '' for x in df['drug_rxnorn_id']]

#Mapping diseases
df['dis'] = mps.mapping(np.asarray(df['condition_meddra_id'], dtype=str),'Disease')

#Edges
edges = sorted(set([(x[0], x[1], round(x[2],2)) for x in df[['cpd','dis', 'PRR']].values if x[0].count('-')==2 and x[1].startswith('MEDDRA:')]))

#Mapping drugs and writing
with open(out_path+'/CPD-cau-DIS/%s.tsv'%source, 'w')  as o:
    o.write('n1\tn2\tPRR\n')
    for e in edges:
        o.write('%s\t%s\t%s\n'%(e[0],e[1],e[2]))
        
sys.stderr.write('Done!\n')
