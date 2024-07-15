root_path = '../../../'
code_path = root_path+'code/embeddings/'

import sys
import os
import pickle
import numpy as np
from tqdm import tqdm
sys.path.insert(0, code_path)
from utils.graph_preprocessing import read_edges

task_id = sys.argv[1]  # <TASK_ID>
filename = sys.argv[2]  # <FILE>

row_uv, columns, paths, opath, is_undirected, is_flip, is_weighted, allow_self_loops = pickle.load(open(filename, 'rb'))[task_id][0]

#--------------------------------------------------------

#This is the operation that is made when more than one path (dataset) is given for the same matrix
"""
Default is "conservative" --> it just update the weights if they were not already provided
(i.e. first come first enter).

Other option is "aggregate" --> the weights are summed
(i.e. the more datasets reporting it the more weight)
"""
merge_operation = 'conservative'
#merge_operation ='aggregate'

#--------------------------------------------------------

for i in range(len(paths)):
    r2v = {r:np.zeros(len(columns)) for r in row_uv} #Should be empty for each dataset.
                                                     #|-->Better performance when updating
    edge_path = paths[i]
    is_w = is_weighted[i]
    is_flp = is_flip[i]

    if is_flp:
        ixs = [1,0]
    else:
        ixs = [0,1]
    c_uv = set(columns)

    for e in tqdm(read_edges(edge_path), desc='Read edges'):

        if not allow_self_loops and e[0] == e[1]: continue

        if e[ixs[0]] in r2v and e[ixs[1]] in c_uv:
            col_ix = np.where(columns==e[ixs[1]])[0][0]

            if is_w is True:
                try:
                    r2v[e[ixs[0]]][col_ix] = float(e[2])
                except (ValueError, IndexError):
                    r2v[e[ixs[0]]][col_ix] = 1
            else:
                r2v[e[ixs[0]]][col_ix] = 1

        if is_undirected:
            if e[ixs[1]] in r2v and e[ixs[0]] in c_uv:
                col_ix = np.where(columns==e[ixs[0]])[0][0]

                if is_w:
                    try:
                        r2v[e[ixs[1]]][col_ix] = float(e[2])
                    except (ValueError, IndexError):
                        r2v[e[ixs[1]]][col_ix] = 1
                else:
                    r2v[e[ixs[1]]][col_ix] = 1

    for r in tqdm(r2v, desc='Write rows'):
        v = r2v[r]

        if os.path.exists(opath+'/%s.npy'%str(r)):

            org_v = np.load(opath+'/%s.npy'%str(r))

            #Rewrite those columns (where the dimensions were already not zero) with the previous value
            if merge_operation == 'conservative':
                msk = org_v>0
                v[msk] = org_v[msk]

            #Summing the weights (the more datasets given an edge the more weight it will have)
            elif merge_operation == 'aggregate':
                v += org_v

        np.save(opath+'/%s.npy'%str(r),v)
