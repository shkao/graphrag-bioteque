import os
import sys
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait()

#Output paths
out_paths = {
'CPD-trt-DIS':out_path+'/CPD-trt-DIS/%s.tsv'%source,
'CPD-cau-DIS':out_path+'/CPD-cau-DIS/%s.tsv'%source,
}
o  = {x:open(y,'w') for x,y in out_paths.items()}
for i in o:
    o[i].write('n1\tn2\n')
    
# Getting mapping: ctd --> inchikey
ctd_inchikey = mps.get_ctd2ikey()

#Reading chemical disease file
m = []
with open("./CTD_chemicals_diseases.tsv",'r') as f:
    a = f.read().splitlines()
    for l in tqdm(a[29:], desc='Reading'):
        h = l.split('\t')
        dg = h[1]
        if dg in ctd_inchikey:
            dg = ctd_inchikey[dg]
        else:
            continue
        dis = h[4]
        ev = h[5]
        if ev == "marker/mechanism":
            ev = 'CAUSES'
        elif ev == "therapeutic":
            ev = 'TREATS'
        else:
            continue
        
        #Stacking results
        m.append([dg,dis,ev])
m = np.asarray(m,dtype=object)

#Parsing diseases
m[:,1] = mps.mapping(m[:,1],'Disease')

#Removing nan values
m = m[~pd.isnull(m[:,1])]

for l in m:
    dg = l[0]
    dis = l[1]
    e = l[2]

    if e == 'TREATS':
        o['CPD-trt-DIS'].write('%s\t%s\n'%(dg,dis))
    elif e == 'CAUSES':
        o['CPD-cau-DIS'].write('%s\t%s\n'%(dg,dis))

#Closing files
for i in o:
    o[i].close()
sys.stderr.write('Done!\n')

