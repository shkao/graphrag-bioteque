import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps
from transform_data import merge_duplicated_cells, drug_sens_stratified_waterfall, quantile_norm

mapping_path = '../../metadata/mappings/'
out_path = '../../graph/raw/CLL-sns-CPD/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#--Mappings
d2ikey = {}
with open(mapping_path+'/CPD/drugcell.tsv', 'r') as f:
    for l in f:
        l = l.rstrip("\n").split("\t")
        d2ikey[int(l[0])] = l[1] if l[1] != 'None' else None

cl2id = {}
with open(mapping_path+'/CLL/drugcell.tsv', 'r') as f:
    for l in f:
        l = l.rstrip("\n").split("\t")
        cl2id[int(l[0])] = l[1] if l[1] != 'None' else None

#-- Read data
df = pd.read_csv('./drugcell_auc.tsv',sep='\t')

#--Map to Bioteque ids
df['cell'] = [cl2id[cl] for cl in df['cell']]
df['drug'] = [d2ikey[dg] for dg in df['drug']]

#--Removing unmapped terms
df = df[(~pd.isnull(df['cell'])) & (~pd.isnull(df['drug']))]

#--Remove duplicates by averaging aucs
df = df.groupby(['cell','drug']).mean().reset_index()

#--Transform into a matrix
df = df.pivot('cell','drug','auc')

#Ternarizing
df = df.transform(drug_sens_stratified_waterfall, axis = 0, ternary=True,
                          min_cls=0.01, max_cls=0.2, apply_uncertainty_at=0.8, min_sens_cutoff=0.9)

#Getting sns
dgs = np.asarray(df.columns)
sns = set([])
for cl , sens in df.iterrows():
    s = dgs[sens ==1]
    sns.update(zip([cl]*len(s),s))

# Writing output
with open(out_path+source+'.tsv','w') as o:
    o.write('n1\tn2\n')
    for r in sorted(sns):
        o.write('%s\t%s\n'%(r[0],r[1]))

sys.stderr.write('Done!\n')
