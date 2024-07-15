import os
import sys
import subprocess
import numpy as np
import csv
from tqdm import tqdm
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait()

# Read DrugBank structures
db2ikey = mps.get_drugbank2ikey()

#Read indications
m = []
f = open("./repodb.csv", "r")
f.readline()
for h in csv.reader(f):

        if h[5] in ('Withdrawn','Suspended','NA'):continue
        dg = h[1]
        if dg in db2ikey:
            dg = db2ikey[dg]
        else:
            continue
        dis = h[3]
        if h[5] in set(['Approved']):
            phase = 4
        elif 'Phase 3' in h[6] :
            phase = 3
        elif 'Phase 2' in h[6]:
            phase = 2
        elif 'Phase 1' in h[6]:
            phase = 1
        elif 'Phase 0' in h[6]:
            phase = 0
        else:
            sys.exit('Unknown phase label: %s'%h[6])

        #Keeping results
        m.append([dg,dis,phase])

#Parsing diseases
m = np.asarray(m,dtype=object)
m[:,1] = mps.parse_diseaseID(m[:,1])

#Writing file
with open(out_path+'/CPD-trt-DIS/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\tphase\n')

    for r in m:
        o.write('%s\t%s\t%i\n'%(r[0],r[1],r[2]))

sys.stderr.write('Done!\n')
