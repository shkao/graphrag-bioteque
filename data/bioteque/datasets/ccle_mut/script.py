import sys
import os
import subprocess
import numpy as np
import pandas as pd
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps
from transform_data import merge_duplicated_cells
out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

"""
We only keep those mutations that are either hotspot or damaging
"""

def get_edges_from_depmap_bool_matrix(df, sample_info_path='./sample_info.csv'):
    
    #--Mapping cls
    cl2id = {}
    unmapped =[]
    sample_info =  pd.read_csv(sample_info_path)
    for cl, cl_id, name in sample_info[['DepMap_ID','RRID','cell_line_name']].values:
        if not pd.isnull(cl_id) and cl_id.startswith('CVCL_'):
            cl2id[cl] = cl_id
        else:
            unmapped.append([cl,name])
    unmapped = np.array(unmapped)
    cl2id.update({unmapped[i][0]:x for i,x in enumerate(mps.cl2ID(unmapped[:,1])) if not pd.isnull(x)})

    df.index = [cl2id[x] if x in cl2id else None for x in df.index.values]
    df = df.drop(index=[None]) #removing unmapped clls

    #Maping genes
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


    # Merge Duplicate Samples 
    df = merge_duplicated_cells(df, 'row', 'max')

    # Merge Duplicate Genes 
    df = merge_duplicated_cells(df, 'column', 'max')

    # Getting edges
    edges = set([])
    gns = np.asarray(df.columns)
    for cl , muts in df.iterrows():
        muts = gns[np.array(muts) == 1]
        edges.update(zip([cl]*len(muts), muts))
        
    return edges

#Running

#--Download the data
subprocess.Popen('./get_data.sh', shell = True).wait()

#--Process mutation matrices and extract edges
edges = set([])
for file in ['./mutations_damaging.csv', './mutations_hotspot.csv']:
    sys.stderr.write("Processing: '%s'...\n"%file)
    edges.update(get_edges_from_depmap_bool_matrix(pd.read_csv(file, index_col=0)))

# Writing output
with open(out_path+'/CLL-mut-GEN/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for r in sorted(edges):
        o.write('%s\t%s\n'%(r[0],r[1]))

sys.stderr.write('Done!\n')
