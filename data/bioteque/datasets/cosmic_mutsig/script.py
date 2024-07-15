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

#---------Parameters---------

min_accuracy = 0.9
signature_count_cutoff = 10 #A minimum of 10 to remove some weak cell-signature associations

#----------------------------

#--Download the data
subprocess.Popen('./get_data.sh', shell = True).wait()

#--Read the data
df = pd.read_excel('./mmc3.xlsx', sheet_name='COSMIC CellLines Signatures', usecols='A:AR', skiprows=[0], header=0)
df['Sample Name'] = df['Sample Name'].astype(str)

#Removing CLs with low accuracy
org_N_CL = df.shape[0]
df = df[df['Accuracy']>=min_accuracy]
print('CLL removed: %i'%(org_N_CL - df.shape[0]))

#Removing signatures that does not appear in any CLL
org_N_sign = df.shape[1]
df = df[list(df.columns[:3])+[x for x in df.columns[3:] if sum(df[x])>0]]
print('Sign removed: %i'%(org_N_sign - df.shape[1]))

#Mapping cls2id
cl2id = dict(zip(df['Sample Name'],mps.cl2ID(df['Sample Name'])))
df['cls'] = [cl2id[cl] for cl in df['Sample Name']]
df = df[~pd.isnull(df['cls'])]
df = df.set_index('cls')

#Removing unwanted columns
df = df[df.columns[3:]]

#Binarizing the dataframe
df[df<signature_count_cutoff] = 0
df[df>=signature_count_cutoff] = 1 

#Getting cll-ass-pwy
cl_ass_pwy = set([])
sig = np.asarray(df.columns)
for cl , data in df.iterrows():
    s = sig[data==1]
    cl_ass_pwy.update(zip([cl]*len(s), s))


with open(out_path+'/CLL-ass-PWY/%s.tsv'%(source),'w') as o:
    o.write('n1\tn2\n')
    for r in sorted(cl_ass_pwy, key = lambda x: x[0]+x[1]):
        o.write('%s\t%s\n'%(r[0],r[1]))
    
sys.stderr.write('Done!\n')

