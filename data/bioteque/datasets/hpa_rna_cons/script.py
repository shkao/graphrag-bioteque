import sys
import os
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps
from transform_data import  merge_duplicated_cells, quantile_norm

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#---------Parameters---------

mx_gns = 250
min_pos_expr= 0.5
min_neg_expr = -0.5

#----------------------------

#Renaming tisues to allow the mapping to Brenda ontology
rename_tis = {'B-cells':'B-lymphocyte', 
              'NK-cells':'natural killer cell', 
              'T-cells':'T-lymphocyte',
             'basal ganglia':'basal ganglion',
             'cervix, uterine': 'uterine cervix',
             'dendritic cells':'dendritic cell',
              'granulocytes':'granulocyte',
             'monocytes':'monocyte',
             'olfactory region':'olfactory organ',
              'total PBMC':'peripheral blood mononuclear cell'
             }
g2g = mps.get_gene2updatedgene()
g2u = mps.get_gene2unip()


#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait()


# Load Data
data = pd.read_csv('./rna_tissue_consensus.tsv', sep='\t')

#Mappings
gns = [g2g[x] if x in g2g else x for x in np.unique(data['Gene name'])]
gns = np.unique([up for g in gns if g in g2u for up in g2u[g]])

tis = np.unique([rename_tis[x] if x in rename_tis else x for x in data['Tissue']])
t2id = dict(zip(tis, mps.tiss2ID(tis)))
tis = np.unique([t2id[x] for x in tis if t2id[x] != None])

#Building matrix
df = np.full((len(tis), len(gns)), np.nan)
for ens, gn, t, score in tqdm(data.values):
    if t in rename_tis:
         t = rename_tis[t]
    t = t2id[t]
    if t is None: continue
    if gn in g2g:
        gn = g2g[gn]
    if gn not in g2u: continue
        
    rix = np.where(tis == t)[0][0]
    for up in g2u[gn]:
        cix = np.where(gns == up)[0][0]
        df[rix,cix] = score
        
df = pd.DataFrame(df, columns=gns, index=tis)

#Standardizing the data (per gene)
scaler = RobustScaler()
df = pd.DataFrame(scaler.fit_transform(df), index = df.index.values, columns = df.columns)

#Getting upr/dwr genes for each tis
tis_upr = set([])
tis_dwr = set([])

gns = np.asarray(df.columns)
for t,r in df.iterrows():
    r = np.asarray(r)
    no_null_ixs = ~pd.isnull(r)
    
    ixs = np.argsort(r[no_null_ixs])
    
    up_ixs = np.arange(len(gns))[no_null_ixs][ixs[-mx_gns:]]
    up_ixs = [x for x in up_ixs if r[x] >= min_pos_expr]
    dw_ixs = np.arange(len(gns))[no_null_ixs][ixs[:mx_gns]]
    dw_ixs = [x for x in dw_ixs if r[x] <= min_neg_expr]
    
    tis_upr.update(zip([t]*len(up_ixs), gns[up_ixs]))
    tis_dwr.update(zip([t]*len(dw_ixs), gns[dw_ixs]))

#upr
with open(out_path+'/TIS-upr-GEN/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for r in sorted(tis_upr):
        o.write('%s\t%s\n'%(r[0],r[1]))

#dwr
with open(out_path+'/TIS-dwr-GEN/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for r in sorted(tis_dwr):
        o.write('%s\t%s\n'%(r[0],r[1]))

sys.stderr.write('Done!\n')

