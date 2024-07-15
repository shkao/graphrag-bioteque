# Disease-localize-Tissue (hetionet)
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

#Reading the data
m = []
with open('./disease-uberon-cooccurrence.tsv','r') as f:
    hd = f.readline().rstrip().split('\t')
    atr_hd = hd[4:-1]
    for l in f:
        h = l.rstrip().split('\t')

        #skipping if the association is not below the p-value
        if float(h[-1]) >= pval_cutoff: continue

        #Stacking the results (not keeping p-value since it will be always ~0)
        m.append([h[0],h[2]]+h[4:-1])

#Mapping tissues and parsing diseases
m = np.asarray(m,dtype=object)

m[:,0] = mps.parse_diseaseID(m[:,0])
m[:,1] = mps.tiss2ID(m[:,1])

#Writing file
with open(out_path+'/TIS-has-DIS/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\t'+'\t'.join(atr_hd)+'\n')

    for r in m:
        d = r[0]
        t = r[1]
        atr = [str(round(float(x),4)) if x !='0.0' else '0.0' for x in (r[2:])]

        #Skipping unmapped data or symptoms that are not in our graph
        if d is None or t is None: continue
        o.write('%s\t%s\t'%(t,d)+'\t'.join(atr)+'\n')

sys.stderr.write('Done!\n')

