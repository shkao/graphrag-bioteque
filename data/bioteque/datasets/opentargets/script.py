#Importing
import os
import sys
import subprocess
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps
out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]
source = 'opentargets'

#--------------Parameters----------º----

sign_cutoff = 0.7
minimum_hits = 100
minimum_score = 0.3

#---------------------------------------

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait() # <-- It may take few minutes!

#protein mappings
sys.path.insert(0,'/aloy/home/afernandez/projects/bioteque/scripts/')
e2up = mps.get_ensembl2up()

#--Disease mappings
disease_metadata_path = './diseases/'
d2d = {}
for x in tqd(os.listdir(disease_metadata_path), desc='getting mapping'):
    if x.endswith('json'):
        v = [json.loads(line) for line in open(disease_metadata_path+x,'r')]
        for i in v:
            ñ,d = i['id'].split('_')
            if ñ in set(['DOID','EFO','HP']):
                d2d[i['id']] = ñ+':'+d
            elif ñ =='Orphanet':
                d2d[i['id']] = 'ORPHA'+':'+d

            else:
                kk  = dict(map(lambda x: x.split(':'),i['dbXRefs']))
                #--Mapping to accepted vocabularies
                for _ in ['DOID','UMLS', 'MESH', 'EFO','HP','ORPHA', 'OMIM']: #prioritize DOID > UMLS > MESH > ...
                    if _ in kk:
                        d2d[i['id']] = _+':'+kk[_]
                        break
                        
#--Reading dataset
p = './associationByOverallIndirect/'
dis_ass = {}
for x in tqdm(os.listdir(p), desc='reading'):
    if x.endswith('json'):
        v = list(map(lambda x: list(x.values()),[json.loads(line) for line in open(p+x,'r')]))
        for d,g,s,c in v:
            if d not in d2d: continue
            d = d2d[d]
            if g not in e2up: continue
            if d not in dis_ass:
                dis_ass[d] = set([])
            dis_ass[d].add((g,s))                

#--Writing associations
with open(out_path+'GEN-ass-DIS/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\tscore\n')
    
    for dis, genes in tqdm(dis_ass.items(), desc='writing'):

        c = 0
        genes = sorted(genes, key = lambda x: x[1],reverse=True)

        for g,score in genes:
            if g not in e2up: continue
            gs = e2up[g]

            if score < minimum_score: 
                break

            elif score >= sign_cutoff:
                for hit in gs:
                    o.write('%s\t%s\t%.3f\n'%(hit,dis,score))
                c+=1

            else:
                if c < minimum_hits:
                    for hit in gs:
                        o.write('%s\t%s\t%.3f\n'%(hit,dis,score))
                    c+=1
