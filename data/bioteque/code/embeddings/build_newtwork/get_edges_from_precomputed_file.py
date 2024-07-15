root_path = '../../../'
code_path = root_path+'code/embeddings/'
graph_edges_path = root_path+'/graph/processed/propagated/'

import sys
import os
import h5py
import numpy as np
import pandas as pd
from bisect import bisect_left
import networkx as nx
import shutil
from tqdm import tqdm
sys.path.insert(0, code_path)
from utils import graph_preprocessing as gpr
from utils.utils import *

def get_row_col_from_file(file, undirected=False, header=True, sep=None):

    if sep is None:
        if file.endswith('.tsv'):
            sep = '\t'
        elif file.endswith('.csv'):
            sep = ','
        else:
            sys.exit('The path :"%s" has not a known suffix (.h5, .tsv, .csv). You must provide a separator!\n'%path)

    rows , cols = set([]), set([])
    with open(file,'r') as f:
        if header:
            next(f)
        for l in f:
            h = l.rstrip().split(sep)
            rows.add(h[0])
            cols.add(h[1])
            if undirected:
                rows.add(h[1])
                cols.add(h[0])
    return sorted(rows), sorted(cols)

# 0) Preparing the input

#--Directories
current_path = sys.argv[1]

edge_opath = current_path+'/edges.h5'
stats_path = current_path+'/stats/'

if not os.path.exists(current_path+'/nodes'):
    os.mkdir(current_path+'/nodes')
if not os.path.exists(stats_path):
    os.mkdir(stats_path)

#--Getting arguments and parameters

args = parse_parameter_file(current_path+'/parameters.txt', graph_edges_path=graph_edges_path)
edge_file = np.array(args['dts_paths']).ravel()[0]
field_separator = args['input_edge_file_sep']
has_header = args['input_edge_file_has_header']
compute_dwpc = np.ravel(args['compute_dwpc'])[0]
is_weighted = np.ravel(args['dts_w'])[0]
min_nodes_in_component = args['min_covered_nodes_in_netw']
dwpc_w = args['damping_weight']

#1) Getting rows and cols
rows, cols = get_row_col_from_file(edge_file, undirected=False)

#2) Removing weak components, compute DWPC (if specified) and updating row/col universe
sys.stderr.write('Removing weak components (if any)...\n')

G = nx.Graph()
nd2dgw = {}
for edge in gpr.read_edges(edge_file, undirected=True):
    if edge[0] == edge[1]: continue
    G.add_edge(edge[0],edge[1])
    if compute_dwpc:
        for nd in edge[:2]:
            if nd not in nd2dgw:
                nd2dgw[nd] = 0
            nd2dgw[nd] += float(edge[2])  if len(edge) > 2 else 1

if nx.number_connected_components(G) == 1:
    nd_uv = set(rows) | set(cols)
    del G
else:
    #The first largest component is always kept.
    #Then, the rest of components are added if they cover at least 5% of both node universes
    u1 = set(rows)
    u2 = set(cols)
    cs = sorted(nx.connected_components(G), key=len, reverse=True)
    nd_uv = set(cs[0])
    for c in cs[1:]:
        if (len(set(c) & u1)/len(u1) >= min_nodes_in_component) and (len(set(c) & u2)/len(u2) >= min_nodes_in_component):
            nd_uv.update(c)

    del G, u1, u2

    #----Removing rows/cols from adj file (if needed)
    rows = [x for x in rows if x in nd_uv]
    cols = [x for x in cols if x in nd_uv]

if compute_dwpc:
    nd2dgw = {x:nd2dgw[x]**-dwpc_w for x in list(nd_uv)}

#3) Writing row/col universe
sys.stderr.write('Writing edge file, metadata and stats...\n')
if args['source'] == args['target']:
    nds = np.unique(rows+cols)
    with open(current_path+'/nodes/%s.txt'%args['source'],'w') as o:
        o.write('\n'.join(nds))
else:
    with open(current_path+'/nodes/%s.txt'%args['source'],'w') as o:
        o.write('\n'.join(rows))
    with open(current_path+'/nodes/%s.txt'%args['target'],'w') as o:
        o.write('\n'.join(cols))
    nds = np.asarray(list(rows)+list(cols))
del rows,cols

#4) Making node2stats and
ids = np.arange(len(nds))+1
with h5py.File(current_path+'/nd2st.h5','w') as o:
    o.create_dataset('nd',data=nds.astype('S'))
    o.create_dataset('id', data= ids)

#5) Making edge file (with mapped ids) --> This will increase efficiency at random walk and metapath steps
nd2id = dict(zip(nds, ids))
egs = []
ws = []
nd_degree = np.zeros(len(ids)) #For statistics
for edge in gpr.read_edges(edge_file):
    n1,n2 = edge[:2]

    #--Skipping if...
    if n1 == n2: continue #Removing self-loops (specially from L1 metapaths that were not checked before) unless they have a tag that differentiate them!
    if n1 not in nd_uv or n2 not in nd_uv: continue #Skipping filtered nodes

    #--Saving weights if any
    if is_weighted:
        if compute_dwpc:
            ws.append(float(edge[2])*nd2dgw[n1]*float(edge[2])*nd2dgw[n2])
        else:
            ws.append(float(edge[2]))
    else:
        if compute_dwpc:
            ws.append(nd2dgw[n1]*nd2dgw[n2])

    #--Mapping nodes and saving edges
    n1, n2 = nd2id[n1], nd2id[n2]
    egs.append([n1,n2])

    #--Kepping nd2degree
    nd_degree[bisect_left(ids,n1)] +=1
    nd_degree[bisect_left(ids,n2)] +=1

#--Keeping unique egs
if args['source'] == args['target']: #If the same node entity, sort each edge so that we can remove repetitions
    egs = np.sort(egs, axis=1)
egs, uixs = np.unique(egs, axis=0, return_index=True)

if len(ws) > 0:
    ws = np.array(ws)[uixs]

#--Writing edges
with h5py.File(current_path+'/edges.h5','w') as o:
    o.create_dataset('edges', data=np.array(egs))
    if ws != []:
        o.create_dataset('weights', data=np.array(ws))
del nd2id, nds, egs, ws

#--Saving nd2degree
with h5py.File(current_path+'/nd2st.h5','a') as o:
    o.create_dataset('degree', data=nd_degree)
del nd_degree

#6) Getting stats
print_info('Getting network stats...\n')
gx = gpr.get_graph_from_edge_list(current_path+'/edges.h5', source_label=args['source'], target_label=args['target'], weighted= is_weighted)

#--writing components
component = np.zeros(len(ids))
for ix,c in enumerate(sorted(nx.connected_components(gx), key=len, reverse=True)):
    ix = ix+1
    for x in c:
        component[bisect_left(ids, int(x))] = ix
with h5py.File(current_path+'/nd2st.h5','a') as o:
    o.create_dataset('component', data=component)
del component

#--Network stats
gpr.network_stats(gx, ofile=stats_path, weighted=is_weighted)
