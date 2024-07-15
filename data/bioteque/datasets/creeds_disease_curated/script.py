import sys
import os
import csv
import subprocess
import numpy as np
from tqdm import tqdm
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait()

#Variables and mappings
gene2unip = mps.get_gene2unip()
gn2updgn = mps.get_gene2updatedgene()

#Output files
out_paths = {
'DIS-upr-GEN':out_path+'/DIS-upr-GEN/%s.tsv'%source,
'DIS-dwr-GEN':out_path+'/DIS-dwr-GEN/%s.tsv'%source,
}
o  = {x:open(y,'w') for x,y in out_paths.items()}
for i in o:
    o[i].write('n1\tn2\n')

#Reading metadata
m = []
with open('./disease_signatures-v1.0.csv','r') as f:
    next(f)
    for h in csv.reader(f):
        ID = h[0]
        umls = h[10]
        if umls == '':continue
        m.append([ID,umls])

m = np.asarray(m,dtype=object)

#Mapping diseases
m[:,1] = mps.parse_diseaseID(m[:,1])

#Reading signatures
id2sign_up = {}
id2sign_dw = {}

with open('./disease_signatures-v1.0.gmt','r') as f:
    for l in tqdm(f):
        h = l.split('\t')
        ID = h[1]
        gns = h[2:]

        #updating gene names
        gns = [x for x in [gn2updgn[x] for x in gns if x in gn2updgn] if x is not None]

        #Mapping
        if h[0].endswith('-up'):
            id2sign_up[ID] = [gn for x in gns if x in gene2unip for gn in gene2unip[x]]
        elif h[0].endswith('-dn'):
            id2sign_dw[ID] = [gn for x in gns if x in gene2unip for gn in gene2unip[x]]

#Writing
for i, dis in m:

    if i in id2sign_up:
        upr = set(id2sign_up[i])
    else:
        upr = set([])
    if i in id2sign_dw:
        dwr = set(id2sign_dw[i])
    else:
        dwr = set([])

    if len(upr) == len(dwr) == 0: continue #Skipping those auxs with unmappend proteins

    #Removing incongruencies
    incongruencies = dwr & upr
    dwr = dwr - incongruencies
    upr = upr - incongruencies

    for gn in upr:
        o['DIS-upr-GEN'].write('%s\t%s\n'%(dis,gn))

    for gn in dwr:
        o['DIS-dwr-GEN'].write('%s\t%s\n'%(dis,gn))

#Clossing files
for i in o:
    o[i].close()

sys.stderr.write('Done!\n')

