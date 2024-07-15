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
import tables
import shutil
from tqdm import tqdm
sys.path.insert(0, code_path)
from utils import graph_preprocessing as gpr
from utils.utils import *

current_path = sys.argv[1]
#-------------------------

#1) Reading and parsing input variables
print_info('Reading and parsing input variables...\n')

#--1.1. Getting arguments form parameter file
args = parse_parameter_file(current_path+'/parameters.txt', graph_edges_path=graph_edges_path)
print_info('\t--> Total metapaths detected: %i\n'%len(args['mpaths']))

#--1.2. Bulding directory system
network_file = current_path+'/network.h5'
edge_file = current_path+'/edges.h5'
stats_path = current_path+'/stats/'
build_directory_system(current_path, n_mpaths=len(args['mpaths']))

# 2) Iterating through each metapath (their matrix will be summed up)
for I in tqdm(range(len(args['mpaths'])), desc='metapaths'):
    metapath = args['mpaths'][I]
    medges = args['medges'][I]
    dts_paths = args['dts_paths'][I]
    dts_flip = args['dts_flip'][I]
    dts_w = args['dts_w'][I]
    compute_dwpc = args['compute_dwpc'][I]
    medges_undirected = args['medges_undirected'][I]
    mxm_order = args['mxm_order'][I]
    damping_weight = args['damping_weight'] #Default: 0.5
    pruning_cutoffs = args['pruning_cutoffs'] #Default: (0.05, 3, 250)
    pruning_method = args['pruning_method'] #Default: "union"
    remove_intra_selfloops = args['remove_intra_selfloops'] #Default: False
    min_nodes_in_component = args['min_covered_nodes_in_netw'] #Default: 0.05
    scratch_path= current_path+'/scratch/edges/mp%i/'%(I+1)
    adj_folder = scratch_path+'/adj'

    print_info('\t--> Working on mpath%i: "%s"\n'%(I+1,mpath2str(metapath)), pre='\n')

    #2.1) Getting and fitting row/col universes
    print_info('%i.1) Getting and fitting row/col universes...\n'%(I+1))
    _rows, _cols = get_rows_and_cols_fitted(dts_paths, dts_flip, medges_undirected, method='union', skip_header=True, allow_self_loops=False)

    #2.2) Get adjancy matrix from edge files
    print_info('%i.2) Getting adjancy matrices for each edge.tsv file...\n'%(I+1))
    transform_edge_tsv_to_adj_h5(dts_paths, _rows, _cols, dts_flip, dts_w, medges_undirected, adj_folder, scratch_path)

    # Adding source/target tags (if specified)
    #------------------------------------------------
    if args['tag_source'] is not None:
        #--Updating source to "__tag__source"
        if not args['source'].startswith(args['tag_source']):
            args['source'] = args['tag_source'] + args['source']
        #--Updating the rows of the first matrix
        with h5py.File(adj_folder+'/m1.h5', 'a') as f:
            _rows = np.array([args['tag_source']+r for r in f['rows'][:].astype(str) if not r.startswith(args['tag_source'])])
            del f['rows']
            f.create_dataset('rows', data=_rows.astype('S'))

    if args['tag_target'] is not None:
        #--Updating source to "__tag__target"
        if not args['target'].startswith(args['tag_target']):
            args['target'] = args['tag_target'] + args['target']
        #--Updating the rows of the first matrix
        with h5py.File(adj_folder+'/m%i.h5'%len(medges), 'a') as f:
            _cols = np.array([args['tag_target']+r for r in f['cols'][:].astype(str) if not r.startswith(args['tag_target'])])
            del f['cols']
            f.create_dataset('cols', data=_cols.astype('S'))
    #--------------------------------------------------

    #2.3) Adding self-loops to undirected metaedges (symmetric matrices)
    print_info('%i.3) Adding self-loops to symmetric matrices (if any)...\n'%(I+1))
    if sum(medges_undirected) > 0 and len(medges) > 1:
        for i, adj_file in enumerate(read_sorted_adj_file_paths(adj_folder)):
            if medges_undirected[i] is True:
                #update_self_loops(adj_file, value=0)
                update_self_loops(adj_file, value=1)

    #2.4) Removing empty rows/cols and refitting their universes
    print_info('%i.4) Removing empty rows/cols...\n'%(I+1))
    remove_empty_rows_and_cols(adj_folder)

    #2.5) Norm and pruning the matrices
    print_info('%i.5) Degree weighting normalization (DWPC)...\n'%(I+1))
    if len(medges) > 1:
        for ix, adj_file in enumerate(read_sorted_adj_file_paths(adj_folder)):
            if np.product(compute_dwpc[ix]) == 0:
                print_info('\t--> Skipping DWPC calculation for the %s dataset (%i-association)\n'%(medges[ix], ix+1))
            else:
                compute_degree_weight(adj_file, damping_w=damping_weight, clear_and_overwrite=True)
    else:
        print_info('\t--> Only one metaedege detected. Skipping DWPC.\n')

    # 2.6) Setting self-loops weight score to 1 (max score)
    #--------------------------------------------------------------------------------------------------
    # IMPORTANT: If the user provide weights the maximum weight must be 1 or this will not make sense!
    #--------------------------------------------------------------------------------------------------
    print_info('%i.6) Setting self-loops weight score to 1 (max score)...\n'%(I+1))
    if sum(medges_undirected) > 0 and len(medges) > 1:
        for i, adj_file in enumerate(read_sorted_adj_file_paths(adj_folder)):
            if medges_undirected[i] is True:
                update_self_loops(adj_file, value=1)

    ##--Getting pruning stats
    dt_stats = get_dataset_pruning_stats(medges, dts_paths, dts_flip, adj_folder)

    #2.7) Matrix multiplication and pruning
    print_info('%i.7) Matrix multiplication and pruning...\n'%(I+1))
    if len(medges) > 1:
        #-- Matrix multiplication
        run_matrix_multiplication(adj_folder, order = mxm_order,
                prune=True, pruning_method=pruning_method, p_neigh = pruning_cutoffs[0], clip= pruning_cutoffs[1:], remove_intra_selfloops=remove_intra_selfloops)
    else:
        print_info('\t--> Only one metaedege detected. No need of multiplication.\n')

    #2.8) Writing network-processing stats
    print_info('%i.8) Writing metapath stats...\n'%(I+1))
    action = 'w' if I ==0  else 'a'
    write_stats(dt_stats, medges, adj_folder, mxm_order, ofile=stats_path+'/metapath_stats.txt', action=action)

#3) Setting the final network
print_info('Setting the final network...\n')

#3.1) Merging final adj matrices (only if more than one metapath was given)
final_adjs = [current_path+'/scratch/edges/mp%i/adj/m%s.h5'%(i+1, ''.join(list(map(str,np.arange(len(args['medges'][i]))+1)))) for i in range(len(args['mpaths']))]
if len(final_adjs) > 1:
    print_info('--> More than one metapath detected. Merging final adjancent matrices...\n', pre='\t')
    aggregate_adj_matrices(final_adjs, network_file, method='union')

#--If only one, just rename it as the final network
else:
    os.rename(final_adjs[0], network_file)

#3.2) Removing weak connected components
print_info('\nChecking connected components...\n')
def overlap(s1,s2):
    return len(s1&s2)/len(s1)

#--Getting Graph to check components
rows,cols = gpr.get_ajd_row_col(network_file)
G = nx.Graph()
for n1,n2,w in gpr.read_edges(network_file, adjancent_matrix=True):
    G.add_edge(n1,n2)

#--Checking components
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
    rows_to_remove = np.where(np.asarray([x not in nd_uv for x in rows]))[0]
    cols_to_remove = np.where(np.asarray([x not in nd_uv for x in cols]))[0]

    if len(rows_to_remove) > 0 or len(cols_to_remove) > 0:
        print_info('--> Removing weak components from the adjancent matrix (min. nodes in a componet: %.3f)\n'%min_nodes_in_component, pre='\t')

    if len(rows_to_remove) > 0:
        remove_ixs_from_adj(network_file, rows_to_remove, 'row')

    if len(cols_to_remove) > 0:
        remove_ixs_from_adj(network_file, cols_to_remove, 'col')

#3.3) Writing row/col universe
rows,cols = gpr.get_ajd_row_col(network_file)
if args['source'] == args['target']:
    nds = np.unique(list(rows)+list(cols))
    with open(current_path+'/nodes/%s.txt'%args['source'],'w') as o:
        o.write('\n'.join(nds))
else:
    with open(current_path+'/nodes/%s.txt'%args['source'],'w') as o:
        o.write('\n'.join(rows))
    with open(current_path+'/nodes/%s.txt'%args['target'],'w') as o:
        o.write('\n'.join(cols))
    nds = np.asarray(list(rows)+list(cols))
del rows,cols

#3.4) Making node2stats and
ids = np.arange(len(nds))+1
with h5py.File(current_path+'/nd2st.h5','w') as o:
    o.create_dataset('nd',data=nds.astype('S'))
    o.create_dataset('id', data= ids)

#3.5) Making edge file (with mapped ids) --> This will increase efficiency at random walk and metapath steps
nd2id = dict(zip(nds, ids))
egs = []
ws = []
nd_degree = np.zeros(len(ids)) #For statistics
for n1,n2,w in gpr.read_edges(network_file, adjancent_matrix=True):

    #--Skipping if...
    if n1 == n2: continue #Removing self-loops (specially from L1 metapaths that were not checked before) unless they have a tag that differentiate them!
    if n1 not in nd_uv or n2 not in nd_uv: continue #Skipping filtered nodes

    n1, n2 = nd2id[n1], nd2id[n2]

    egs.append([n1,n2])
    ws.append(w)

    #--Kepping nd2degree
    nd_degree[bisect_left(ids,n1)] +=1
    nd_degree[bisect_left(ids,n2)] +=1

#Keeping unique egs
if args['source'] == args['target']: #If the same node entity, sort each edge so that we can remove repetitions
    egs = np.sort(egs, axis=1)
egs, uixs = np.unique(egs, axis=0, return_index=True)
ws = np.array(ws)[uixs]

#Writing edges
with h5py.File(edge_file,'w') as o:
    o.create_dataset('edges', data=np.array(egs))
    if len(medges) > 1:
        o.create_dataset('weights', data=np.array(ws))
del nd2id, nds, egs, ws

#--Saving nd2degree
with h5py.File(current_path+'/nd2st.h5','a') as o:
    o.create_dataset('degree', data=nd_degree)
del nd_degree

#4) Getting stats
print_info('Getting network stats...\n')
gx = gpr.get_graph_from_edge_list(edge_file, source_label=args['source'], target_label=args['target'], weighted= True if len(medges) > 1 else False)

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
gpr.network_stats(gx, ofile=stats_path, weighted=True if len(medges) > 1 else False)

#5) Cleaning up
print_info('Cleaning up...\n')
#--Remove scratch folders
for I in tqdm(range(len(args['mpaths'])), desc='Cleaning scratch'):
    scratch_path= current_path+'/scratch/edges/mp%i'%(I+1)
    if os.path.exists(scratch_path):
        shutil.rmtree(scratch_path)

#Removing network
if os.path.exists(current_path+'/network.h5'):
    os.remove(current_path+'/network.h5')
