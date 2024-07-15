import sys
import os
import subprocess
import numpy as np
import pandas as pd
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps
out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#----------Parameters--------------

percentile_cutoff = 95

#----------------------------------

#Download the data
subprocess.Popen('./get_data.sh', shell = True).wait()

#Reading NCI60 zscore data
df = pd.read_excel('./output/DTP_NCI60_ZSCORE.xlsx', sheet_name='all', skiprows=8, usecols='A,G:BN', index_col=0, na_values='na')

# Binarizing by taking the 5% most sensitivte cpds for each cell line 
m_binary = []
for cl in df:
    s = np.array(df[cl])
    v = np.zeros(df.shape[0])
    cutoff = np.nanpercentile(s , percentile_cutoff)
    v[s >= cutoff] = 1
    m_binary.append(v)

df = pd.DataFrame(m_binary, index=df.columns, columns= df.index.values).T
#---> While taking 5% most sensitive cell lines for each cpd is also an option, with only 60 cell lines the CPD-CLL associations may not be robust...

# Mapping

m = []
dgs = []
nci2ikey = mps.get_nci2ikey()
df.index = [nci2ikey[str(ix)] if str(ix) in nci2ikey else None for ix in df.index.values]
df = df.iloc[~pd.isnull(df.index.values)]
df.columns = [x.split(':')[-1] for x in df.columns]

#Mapping CCls
df.columns = mps.cl2ID(df.columns)

#Removing Unmapped CCLs
df = df[[x for x in df.columns if not pd.isnull(x)]]

#--Getting CPD-CLL sensitive pairs
cll_sns_cpd = set([])
cls = np.asarray(df.columns)
for dg, data in df.iterrows():
    s = cls[np.asarray(data==1)]
    cll_sns_cpd.update(zip(s, [dg]*len(s)))

#Writing file
with open(out_path+'/CLL-sns-CPD/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for r in sorted(cll_sns_cpd):
        o.write('%s\t%s\n'%(r[0],r[1]))
        
sys.stderr.write('Done!\n')
