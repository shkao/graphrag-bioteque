import sys
import os
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps
from transform_data import drug_sens_stratified_waterfall

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait()

#Drug2id
d2i = mps.get_prism2ikey()
        
#Getting drug, cell
m= []
auc_column_ix = 8
with open('./secondary-screen-dose-response-curve-parameters.csv', 'r') as f:
    f.readline()
    for l in f:
        r = l.rstrip('\n').split(',')
        #--Mapping Drugs
        if r[0] not in d2i: continue
        if r[2] == 'NA': continue
        m.append([d2i[r[0]], r[2].split('_')[0], r[auc_column_ix]])
df = pd.DataFrame(m, columns = ['d','c','s'])
df["s"] = pd.to_numeric(df["s"])

#--Averaging repetitions
df = df.groupby(['d','c'], as_index=False).mean()

#--Mapping CLs
c2i = dict(zip(np.unique(df['c']), mps.cl2ID(np.unique(df['c']))))
c2i['TT'] = 'CVCL_1774' #Manually curated mapping
df['c'] = [c2i[c] for c in df['c']]
df = df.iloc[list(~pd.isnull(df['c']))].reset_index(drop=True)

#Building df_sns 
cols = np.unique(df['d'])
rows = np.unique(df['c'])

m = np.full((len(rows), len(cols)), np.nan)
for r in tqdm(df.values):
    cix = np.where(cols==r[0])[0][0]
    rix = np.where(rows==r[1])[0][0]
    m[rix,cix] = r[2]
df_sns = pd.DataFrame(m, index=rows, columns=cols)

#Binarizing
df_sns = df_sns.transform(drug_sens_stratified_waterfall, axis = 0, ternary=False,
                          min_cls=0.01, max_cls=0.2, apply_uncertainty_at=0.8, min_sens_cutoff=0.9)

#Getting sns
dgs = np.asarray(df_sns.columns)
sns = set([])
for cl , sens in df_sns.iterrows():
    s = dgs[sens ==1]
    sns.update(zip([cl]*len(s),s))

# Writing output
with open(out_path+'/CLL-sns-CPD/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for r in sorted(sns):
        o.write('%s\t%s\n'%(r[0],r[1]))


sys.stderr.write('Done!\n')

