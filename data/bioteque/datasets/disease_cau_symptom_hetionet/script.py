# Disease-has-Symptom (hetionet)
import os
import sys
import subprocess
import numpy as np
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#-------Parameters--------

pval_cutoff = 0.005 #As used by Hetionet

#-------------------------

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait()

#Reading our symptoms
symptoms = set([])
with open('./mesh_sign_and_symptoms_D012816.tsv','r') as f:
    next(f)
    for l in f:
        symptoms.add(l.rstrip('\n').split('\t')[0])

#Reading their data
m = []

with open('./disease-symptom-cooccurrence.tsv','r') as f:
    hd = f.readline().rstrip().split('\t')
    atr_hd = hd[4:-1]
    for l in f:
        h = l.rstrip().split('\t')
        
        #skipping if not covered by mesh sign and symptoms
        if h[2] not in symptoms: continue
        
        #skipping if the association is not below the p-value
        if float(h[-1]) >= pval_cutoff: continue
        
        dis = h[0]
        symp = 'MESH:'+h[2]
        m.append([dis,symp]+h[4:-1])

#Writing file
with open(out_path+'/DIS-cau-DIS/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\t'+'\t'.join(atr_hd)+'\n')
    for r in m:
        d = r[0]
        s = r[1]
        atr = [str(round(float(x),4)) if x !='0.0' else '0.0' for x in (r[2:])]
        o.write('%s\t%s\t'%(d,s)+'\t'.join(atr)+'\n')
sys.stderr.write('Done!\n')

