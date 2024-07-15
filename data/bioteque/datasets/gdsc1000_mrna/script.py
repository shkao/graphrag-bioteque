import sys
import os
import subprocess
import numpy as np
import pandas as pd
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps
from transform_data import  merge_duplicated_cells
from sklearn.preprocessing import RobustScaler

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#-----Parameters------

min_screened_cls = 0.05
max_gns = 250

#---------------------

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait()

# Load Data
df = pd.read_csv('./sanger1018_brainarray_ensemblgene_rma.txt', sep='\t', index_col=0)

# Drop Any Genes That Have Zero Expression Across 95% Of The Samples
df.replace(0, np.nan, inplace=True)
df.dropna(thresh=(min_screened_cls*df.shape[1]), axis=0, inplace=True)
df.replace(np.nan, 0, inplace=True)
df = df.T

# Mapping CLs
df_cl = pd.read_excel('./Cell_Lines_Details.xlsx')
cl2n = dict(zip([str(int(x)) for x in df_cl['COSMIC identifier'] if not pd.isnull(x)],list(df_cl['Sample Name'].astype(str))))
cl2id = dict(zip(list(cl2n.values()),mps.cl2ID(list(cl2n.values()))))

m_cls = set(df.index.values)
df = df.loc[list(m_cls & set(cl2n))]
df.index = [cl2id[cl2n[x]] for x in df.index.values]
df = df.drop(index=[None]) #removing unmapped clls

# Mapping Genes
e2u = {}
e2u = mps.get_ensembl2up()

df = df[list(set(df.columns)&set(e2u))]

m = []
ixs = []
for gn in df.columns:
    r = df[gn]
    for g in e2u[gn]:
        ixs.append(g)
        m.append(list(r))
df = pd.DataFrame(m,index=ixs, columns=df.index.values).T

# Merge Duplicate Samples By Rows (by taking the mean)
df = merge_duplicated_cells(df, 'row', 'mean')

# Merge Duplicate Genes By Columns
df = merge_duplicated_cells(df, 'column', 'mean')

#Standardizing the data
#--->The data is already RMA normalized (log2 + quantile normalization) so we cand directly standarize them
scaler = RobustScaler()
df = pd.DataFrame(scaler.fit_transform(df), index = df.index.values, columns = df.columns)

#Getting upr/dwr genes per cell line
gns = np.asarray(df.columns)

upr,dwr = set([]), set([])
for cl , expr in df.iterrows():
    expr = np.asarray(expr)
    ixs = np.argsort(expr)
    dwg = expr[ixs[:max_gns]]
    upg = expr[ixs[-max_gns:]]
    upr.update(zip([cl]*max_gns, gns[ixs[-max_gns:]]))
    dwr.update(zip([cl]*max_gns, gns[ixs[:max_gns]]))

# Writing output
if not os.path.exists(out_path+'/CLL-upr-GEN'):
    os.mkdir(out_path+'/CLL-upr-GEN')

if not os.path.exists(out_path+'/CLL-dwr-GEN'):
    os.mkdir(out_path+'/CLL-dwr-GEN')

#upr
with open(out_path+'/CLL-upr-GEN/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for r in sorted(upr):
        o.write('%s\t%s\n'%(r[0],r[1]))

#dwr
with open(out_path+'/CLL-dwr-GEN/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for r in sorted(dwr):
           o.write('%s\t%s\n'%(r[0],r[1]))

sys.stderr.write('Done!\n')
