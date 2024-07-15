import sys
import os
import pandas as pd
import numpy as np
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps
from transform_data import merge_duplicated_cells
out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#----- Parameters ------
mx_gns = 250
min_pos = 0.5
min_neg = -0.5
#-----------------------

# !! IMPORTANT: You need to donwload Table S2 ("mmc2.xlsx") from https://doi.org/10.1016/j.cell.2019.12.023
sys.stderr.write('Reading the excel table (may take few minutes)...')
sys.stderr.flush()
df = pd.read_excel('./mmc2.xlsx', sheet_name='Normalized Protein Expression', usecols='A:PJ').dropna(axis='columns', how='all')
df = df.set_index('Uniprot_Acc')
df = df[df.columns[47:]]
sys.stderr.write('done!\n')

#Treating CLS
df.columns = [x.split('_')[0] for x in df.columns]

#--Mapping CLs
cl2id = dict(zip(df.columns, mps.cl2ID(df.columns)))
df.columns = [cl2id[c] for c in df.columns]
df = df[[x for x in df.columns if not pd.isnull(x)]]

#--Merge Duplicate Genes By Columns
df = merge_duplicated_cells(df, 'column', 'mean')

#Treating gns
good_gns = sorted(set(df.index.values) & set(mps.get_human_reviewed_uniprot()))
df = df.loc[good_gns]

cl_pab_gn = set([])
cl_pdf_gn = set([])

#Getting edges
genes = np.asarray(df.index.values)
for cl in df.columns:
    ixs = ~pd.isnull(df[cl])
    gns = genes[ixs]
    v = np.asarray(df[cl])[ixs]
    pos_ix = np.argsort(v)[-mx_gns:]
    neg_ix = np.argsort(v)[:mx_gns]
    
    pos_ix = [i for i in pos_ix if v[i] >= min_pos]
    neg_ix = [i for i in neg_ix if v[i] <= min_neg]
    
    pos = gns[pos_ix]
    neg = gns[neg_ix]
    
    cl_pab_gn.update(zip([cl]*len(pos), pos))
    cl_pdf_gn.update(zip([cl]*len(neg), neg))

# Writing output

#pab
with open(out_path+'/CLL-pab-GEN/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for r in sorted(cl_pab_gn):
        o.write('%s\t%s\n'%(r[0],r[1]))

#pdf
with open(out_path+'/CLL-pdf-GEN/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for r in sorted(cl_pdf_gn):
        o.write('%s\t%s\n'%(r[0],r[1]))

sys.stderr.write('Done!\n')
