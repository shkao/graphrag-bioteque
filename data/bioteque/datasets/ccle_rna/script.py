import sys
import os
import subprocess
import numpy as np
import pandas as pd
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps
from transform_data import merge_duplicated_cells, quantile_norm
from sklearn.preprocessing import RobustScaler
out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#----Parameters-----

max_gns = 250

#-------------------

#Download the data
subprocess.Popen('./get_data.sh', shell = True).wait()

# Load Data
df = pd.read_csv('./expression.csv',index_col=0)

# Drop Any Genes That Have Zero Expression Across 95% Of The Samples
df.replace(0, np.nan, inplace=True)
df.dropna(thresh=(0.05*df.shape[0]), axis=1, inplace=True)
df.replace(np.nan, 0, inplace=True)

# Mapping cls
cl2id = {}
unmapped =[]
sample_info =  pd.read_csv('./sample_info.csv')
for cl, cl_id, name in sample_info[['DepMap_ID','RRID','cell_line_name']].values:
    if not pd.isnull(cl_id) and cl_id.startswith('CVCL_'):
        cl2id[cl] = cl_id
    else:
        unmapped.append([cl,name])
unmapped = np.array(unmapped)
cl2id.update({unmapped[i][0]:x for i,x in enumerate(mps.cl2ID(unmapped[:,1])) if not pd.isnull(x)})

df.index = [cl2id[x] if x in cl2id else None for x in df.index.values]
df = df.drop(index=[None]) #removing unmapped clls

# Maping genes
g2g = mps.get_gene2updatedgene()
g2u = mps.get_gene2unip()
gid2up = mps.get_geneID2unip()

m = []
ix = []
for g,r in df.T.iterrows():
    g,gid = g.split(' ')
    gid = gid.strip('()')

    #--Mapping
    if g in g2g:
        g = g2g[g]

    if g in g2u:
        ups = g2u[g]
    elif gid in gid2up:
        ups = gid2up[gid]
    else:
        continue

    for up in ups:
        ix.append(up)
        m.append(list(r))
df = pd.DataFrame(m, index=ix, columns=df.index.values).T

# Merge Duplicate Samples By Rows (by taking the mean)
df = merge_duplicated_cells(df, 'row', 'mean')

# Merge Duplicate Genes By Columns
df = merge_duplicated_cells(df, 'column', 'mean')

#Quantile normalization for the columns
df = quantile_norm(df)

# Standardizing the data
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
