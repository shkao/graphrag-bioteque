root_path = '../../../'
code_path = root_path+'code/embeddings/'
singularity_image = root_path+'/programs/ubuntu-py.img'
hpc_path = root_path+'/programs/hpc/'

import sys
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
import re
import tables
from itertools import combinations
import copy
import shutil
import datetime
sys.path.insert(0, hpc_path)
from hpc import HPC
from config import config as cluster_config
sys.path.insert(1, code_path)
from utils import graph_preprocessing as gpr
from utils.utils import medges2mpath

def print_info(s, pre=''):
    sys.stderr.write(pre+datetime.datetime.now().strftime("%Y-%m-%d|%H:%M:%S") +'-INFO: %s'%s)
    sys.stderr.flush()

def get_rows_cols(path):
    r = set([])
    c = set([])
    if path.endswith('.h5'):
        with h5py.File(path, 'r') as f:
            r.update(f['rows'][:].astype(str))
            c.update(f['cols'][:].astype(str))
    else:
        if path.endswith('.tsv'):
            sep = '\t'
        elif path.endswith('.csv'):
            sep = ','

        with open(path,'r') as f:
            for l in f:
                if l.startswith('n1'):continue
                h = l.rstrip('\n').split(sep)
                r.add(h[0])
                c.add(h[1])

    return sorted(r), sorted(c)

def recover_original_network_from_emb_path(current_path, undirected=False):

    #--Getting mapped edges, rows and cols
    with h5py.File(current_path+'/nd2st.h5','r') as f:
        id2nd = dict(zip(f['id'][:].astype(str), f['nd'][:].astype(str)))

    with open(current_path+'/edges.tsv', 'r') as f, open(current_path+'/_edges.tsv','w') as o:
        rows, cols = set([]), set([])
        for l in f:
            h = l.rstrip('\n').split('\t')
            h[0], h[1] = id2nd[h[0]], id2nd[h[1]]
            rows.add(h[0])
            cols.add(h[1])
            o.write('\t'.join(h)+'\n')

    if undirected:
        rows = cols = np.asarray(sorted(rows|cols))
    else:
        rows, cols = np.asarray(sorted(rows)), np.asarray(sorted(cols))

    scratch_path = current_path+'/tmp'
    if not os.path.exists(scratch_path): os.mkdir(scratch_path)
    transform_edge_tsv_to_adj_h5(dts_paths=[current_path+'/_edges.tsv'], _rows=[rows], _cols=[cols], dts_flip=[False],
                                 dts_w=[True], dts_undirected=[undirected], opath=current_path, scratch_path=scratch_path)

    os.rename(current_path+'/m1.h5', current_path+'/network.h5')
    shutil.rmtree(scratch_path)
    os.remove(current_path+'/_edges.tsv')

def read_sorted_adj_file_paths(p):
    return sorted([p+'/%s'%x for x in os.listdir(p) if x.endswith('.h5')])

def read_h5_chunk(path, chunk=None, axis='col'):
    with h5py.File(path,'r') as f:
        rows = f['rows'][:].astype(str)
        cols = f['cols'][:].astype(str)

        if chunk:
            if axis == 'col':
                return pd.DataFrame(f['m'][:, chunk[0]:chunk[1]], index = rows, columns=cols[chunk[0]:chunk[1]])
            else:
                return pd.DataFrame(f['m'][chunk[0]:chunk[1],:], index= rows[chunk[0]:chunk[1]], columns = cols)

#------------------------------------------------
# Get rows/cols fitted for matrix multiplication
#------------------------------------------------
def get_rows_and_cols_fitted(dts_paths, dts_flip, dts_undirected, method='union', skip_header=True, allow_self_loops=False):

    #Fit universes
    rows, cols = [], []

    #--Iterating through metaedges
    for ix1 in range(len(dts_paths)):
        dts = dts_paths[ix1]
        flips = dts_flip[ix1]
        undirected = dts_undirected[ix1]

        if type(dts) == str:
            dts = [dts]
        if type(flips) == bool:
            flips = [flips]*len(dts)
        if type(undirected) == bool:
            undirected = [undirected]*len(dts)

        #--Iterating through dts in metadege
        _r, _c = set([]), set([])
        for ix2 in range(len(dts)):
            dt = dts[ix2]
            flip = flips[ix2]
            is_undirected = undirected[ix2]

            if flip: order = [1,0]
            else: order = [0,1]

            #--Reading edges
            with open(dt,'r') as f:
                lbs = [set([]),set([])]
                if skip_header:
                    f.readline()
                for l in f:
                    e = l.rstrip('\n').split('\t')[:2]
                    if not allow_self_loops and e[0] == e[1]: continue #This will prevent rows and cols that only exists due to a self-loop
                    lbs[0].add(e[0])
                    lbs[1].add(e[1])
                    if is_undirected:
                        lbs[0].add(e[1])
                        lbs[1].add(e[0])

            #--Merge edges at metaedge level
            if method == 'union':
                _r = set.union(_r, set(lbs[order[0]]))
                _c = set.union(_c, set(lbs[order[1]]))
            elif method == 'intersection':
                if len(_r) == 0:
                    _r = set(lbs[:,order[0]])
                    _c = set(lbs[:,order[1]])
                else:
                    _r = set.intersection(_r, set(lbs[:,order[0]]))
                    _c = set.intersection(_c, set(lbs[:,order[1]]))
            else:
                sys.exit('Invalid method "%s". Supported methods are: "union", "intersection".\n'%method)

        rows.append(np.asarray(sorted(_r)))
        cols.append(np.asarray(sorted(_c)))

    #Fitting universes to allow matrix multiplication
    for i in range(len(rows)-1):
        uv = np.asarray(sorted(set(cols[i]) & set(rows[i+1])))
        cols[i] = uv
        rows[i+1] = uv

    return rows, cols

#----------------------------------------------
# Transform edge.tsv file to adjancent matrix
#----------------------------------------------
def transform_edge_tsv_to_adj_h5(dts_paths, _rows, _cols, dts_flip, dts_w, dts_undirected, opath, scratch_path,
                                 allow_self_loops=False, chunk=1000, atom= tables.Float64Atom(),
                                write_row_function_path  = code_path+'/bq2emb_scripts/_write_rows.py'):

    row_opath =  scratch_path+'/_r/'
    if not os.path.exists(opath):
        os.mkdir(opath)

    #--Iterating through datasets
    for ix in tqdm(range(len(dts_paths)),desc='processing datasets'):

        #--Directory
        if os.path.exists(row_opath):
            shutil.rmtree(row_opath)
        os.mkdir(row_opath)

        #--Variables
        dt_paths = dts_paths[ix]
        is_flip = dts_flip[ix]
        is_weight = dts_w[ix]
        is_undirected = dts_undirected[ix] #This depends on the metaedge and not to the file so the same label
                                           #should be used for every file associated with it.

        if type(dt_paths) == str:
            dt_paths = [dt_paths]
        if type(is_flip) == bool:
            is_flip = [is_flip]*len(dt_paths)
        if type(is_weight) == bool:
            is_weight = [is_weight]*len(dt_paths)

        #--Matrix dimensions and labels
        cols = np.asarray(_cols[ix])
        rows = np.asarray(_rows[ix])
        shape = (len(rows), len(cols))

        #--Building elements to iterate in the cluster
        elements = [[
                set(rows[i:i+chunk]),
                cols,
                dt_paths,
                row_opath,
                is_undirected,
                is_flip,
                is_weight,
                allow_self_loops]
                for i in np.arange(0,len(rows),chunk)]

        #-- Calling cluster
        if len(cols) > 20000:
            cpu = 2
        else:
            cpu = 1
        cluster = HPC(**cluster_config)
        cluster_params = {}
        cluster_params['job_name'] = 'get_adj.h5'
        cluster_params["jobdir"] = scratch_path+'/scratch'
        cluster_params['cpu'] = cpu
        cluster_params["wait"] = True
        cluster_params["elements"] = elements
        cluster_params["num_jobs"] = len(elements)

        command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
        singularity_image,
        write_row_function_path)
        cluster.submitMultiJob(command, **cluster_params)

        #3) Iterating for each row and writing results
        with tables.open_file(opath+'/m%i.h5'%(ix+1), 'w') as h5:
            out_m = h5.create_carray(h5.root, 'm', atom, shape, chunkshape=(1, len(cols)))

            r_ix = 0
            for row in tqdm(rows, desc='Writing rows (Adj%i)'%(ix+1), leave=False):
                out_m[r_ix:r_ix+1, 0:len(cols)] = np.load(row_opath+'/%s.npy'%row)
                r_ix+=1

            #Writing rows and cols
            h5.create_array(h5.root,'rows',obj=rows.astype('S'))
            h5.create_array(h5.root,'cols',obj=cols.astype('S'))

#--------------------------
# Removing empty rows/cols
#--------------------------
def get_empty_lines(adj_file, axis='row', chunk=1000):

    with h5py.File(adj_file,'r') as f:

        ixs = []

        if axis == 'row':
            for i in tqdm(np.arange(0,f['m'].shape[0], chunk), leave=False):
                v = np.argwhere(np.all(f['m'][i:i+chunk,:][..., :] == 0, axis=1)).ravel()
                if len(v) > 0:
                    v+=i
                    ixs.extend(v)

        elif axis == 'col':
            for i in tqdm(np.arange(0,f['m'].shape[1], chunk), leave=False):
                v = np.argwhere(np.all(f['m'][:,i:i+chunk][..., :] == 0, axis=0)).ravel()
                if len(v) > 0:
                    v+=i
                    ixs.extend(v)
    return np.asarray(ixs)

def remove_ixs_from_adj(adj_file, _ixs, axis, batch=1000):

    ixs = np.unique(_ixs)

    with tables.open_file(adj_file, 'a') as h5:

        mshape = list(h5.root.m.shape)

        if axis == 'row':
            mshape[0] = mshape[0]- len(ixs)
            chunkshape = (batch, mshape[1])

            out_m = h5.create_carray(h5.root, '_m', h5.root.m.atom, mshape, chunkshape=chunkshape)

            rows_to_keep = []
            #--Iterating and writing new matrix
            for i_batch in tqdm(np.arange(0, h5.root.m.shape[0], batch), leave=False):

                v = h5.root.m[i_batch:i_batch+batch, :]
                v_ixs = np.arange(v.shape[0])+ i_batch
                ixs_to_keep = ~np.in1d(v_ixs, ixs)
                v = v[ixs_to_keep]

                out_m[len(rows_to_keep):len(rows_to_keep)+v.shape[0],:] = v
                rows_to_keep.extend(v_ixs[ixs_to_keep])

            #--Getting new labels
            new_rows = np.asarray(h5.root.rows)[rows_to_keep]
            new_cols = np.asarray(h5.root.cols)

        if axis == 'col':
            mshape[1] = mshape[1]- len(ixs)
            chunkshape = (mshape[0], batch)

            out_m = h5.create_carray(h5.root, '_m', h5.root.m.atom, mshape, chunkshape=chunkshape)

            cols_to_keep = []

            #--Iterating and writing new matrix
            for i_batch in tqdm(np.arange(0, h5.root.m.shape[1], batch), leave=False):

                v = h5.root.m[:, i_batch:i_batch+batch]
                v_ixs = np.arange(v.shape[1])+ i_batch
                ixs_to_keep = ~np.in1d(v_ixs, ixs)
                v = v[:, ixs_to_keep]

                out_m[:, len(cols_to_keep):len(cols_to_keep)+v.shape[1]] = v
                cols_to_keep.extend(v_ixs[ixs_to_keep])

            #--Getting new labels
            new_rows = np.asarray(h5.root.rows)
            new_cols = np.asarray(h5.root.cols)[cols_to_keep]

    #Renaming, updating and removing old data
    with h5py.File(adj_file, 'a') as h5:
        #--Update matrix
        del h5['m']
        h5['m'] = h5['_m']
        del h5['_m']
        #--Update rows
        del h5['rows']
        h5.create_dataset('rows', data=new_rows.astype('S'))
        #--Update cols
        del h5['cols']
        h5.create_dataset('cols', data=new_cols.astype('S'))

def remove_empty_rows_and_cols(adj_paths):

    if type(adj_paths) == str:
        if os.path.isdir(adj_paths):
            adj_paths = read_sorted_adj_file_paths(adj_paths)
        else:
            adj_paths = [adj_paths]

    #--Getting rows and cols that are all zero (need to be removed)
    ixs_to_remove = []
    for adj_file in adj_paths:
        ixs_to_remove.append([get_empty_lines(adj_file, 'row'), get_empty_lines(adj_file, 'col')])

    #--Fitting the ixs respecting the universes (cols[n] == rows[n+1])
    for i in range(len(ixs_to_remove)-1):
        uv = np.unique(list(ixs_to_remove[i][1])+ list(ixs_to_remove[i+1][0]))
        ixs_to_remove[i][1] = uv
        ixs_to_remove[i+1][0] = uv

    #--Removing empty lines
    for i in tqdm(range(len(adj_paths)), desc='datasets', leave=False):
        adj_file = adj_paths[i]

        #--Rows
        ixs = ixs_to_remove[i][0]
        if len(ixs) > 0:
            remove_ixs_from_adj(adj_file, ixs, 'row')

        #--Cols
        ixs = ixs_to_remove[i][1]
        if len(ixs) > 0:
            remove_ixs_from_adj(adj_file, ixs, 'col')


#--------------------------
# Update self loops
#--------------------------
def update_self_loops(adj_file, value=0, batch=1000):

    #Check if there are self loops. If not do nothing
    with h5py.File(adj_file, 'r') as f:
        if len(set(f['rows'][:].astype(str)) & set(f['cols'][:].astype(str))) == 0:
            print('%s can not have self-loops!\n'%adj_file)
            return None

    with tables.open_file(adj_file, 'a') as h5:

        _rows = np.array(h5.root.rows, dtype=str)
        cols = np.array(h5.root.cols, dtype=str)
        col_uv = set(cols)
        mshape = list(h5.root.m.shape)
        chunkshape = (batch, mshape[1])
        out_m = h5.create_carray(h5.root, '_m', h5.root.m.atom, mshape, chunkshape=chunkshape)

        for i_batch in tqdm(np.arange(0, h5.root.m.shape[0], batch)):
            rows = _rows[i_batch:i_batch+batch]
            row_uv = set(rows)
            v = h5.root.m[i_batch:i_batch+batch, :]

            #Removing edge/weights froms self loops (i.e. when row_label == col_label)
            for label in row_uv & col_uv:
                rix = np.where(rows == label)[0]
                cix = np.where(cols  == label)[0]
                v[rix,cix] = value

            out_m[i_batch:i_batch+batch, :] = v

    #Renaming, updating and removing old data
    with h5py.File(adj_file, 'a') as h5:
        #--Update matrix
        del h5['m']
        h5['m'] = h5['_m']
        del h5['_m']

#--------------------
# Degree_weighted_norm matrix
#--------------------
def degree_weight_in_axis(m, w = 0.5, axis='col'):
    if axis == 'col':
        return np.nan_to_num(m* m.sum(0)**-w)

    elif axis == 'row':
        return np.nan_to_num(m.T* m.sum(1)**-w).T

def compute_degree_weight(adj_file, damping_w = 0.5, batch= 1000,
             atom = tables.Float64Atom(),clear_and_overwrite=True):

    with tables.open_file(adj_file, 'a') as h5:
        dts_in_h5 = set([x.name for x in list(h5.root)])

        rows = np.array(h5.root.rows, dtype=str)
        cols = np.array(h5.root.cols, dtype=str)
        mshape = h5.root.m.shape

        #1) Preparing m for n1 (m4n1)
        #print_info('Working on m4n1...\n')

        #Columns
        chunkshape = (len(rows), batch)
        if 'col_w' in dts_in_h5:
            h5.root.col_w._f_remove()

        out_m = h5.create_carray(h5.root, 'col_w', atom, mshape, chunkshape=chunkshape)
        for i_batch in tqdm(np.arange(0,len(cols), batch), desc='degree weighting cols', leave=False):
            out_m[:, i_batch:i_batch+batch] = degree_weight_in_axis(h5.root.m[:, i_batch:i_batch+batch], w=damping_w, axis='col')

        #Rows
        chunkshape = (batch, len(cols))
        if 'row_w' in dts_in_h5:
            h5.root.row_w._f_remove()

        out_m = h5.create_carray(h5.root, 'row_w', atom, mshape, chunkshape=chunkshape)
        for i_batch in tqdm(np.arange(0,len(rows), batch), desc='degree weighting rows', leave=False):

            out_m[i_batch:i_batch+batch, :] = degree_weight_in_axis(h5.root.m[i_batch:i_batch+batch, :], w=damping_w, axis='row')

        #Merge degree weightings
        if 'm_norm' in dts_in_h5:
            h5.root.m_norm._f_remove()

        out_m = h5.create_carray(h5.root, 'm_norm', atom, mshape, chunkshape=(10000,10000))
        for rix in tqdm(np.arange(0,len(rows), 10000), desc='merging', leave=False):
            for cix in np.arange(0,len(cols), 10000):
                out_m[rix:rix+10000, cix:cix+10000] = h5.root.row_w[rix:rix+10000, cix:cix+10000] * h5.root.col_w[rix:rix+10000, cix:cix+10000]

    if clear_and_overwrite:
       #Renaming, updating and removing added data
        with h5py.File(adj_file, 'a') as h5:
            #--Removing old data
            for file in ['col_w', 'row_w']:
                if file in h5.keys():
                    del h5[file]
            #--Updating m
            del h5['m']
            h5['m'] = h5['m_norm']
            del h5['m_norm']

#--------------------
# Prune/Norm matrix
#--------------------
def prune(m, n_neigh, axis='row', rank_method=None):

    from scipy.stats import rankdata

    new_m = np.zeros(m.shape)

    if axis == 'row':
        it = (x for x in m)
    elif axis == 'col':
        it = (x for x in m.T)
        new_m = new_m.T

    #Prun
    for i, r in enumerate(it):

        #--Getting neigh based on cutoff
        sorted_r_ixs = np.argsort(r)[::-1]

        if r[sorted_r_ixs[n_neigh-1]] == 0:
            ixs = np.where(r)[0]
        else:
            ixs = sorted_r_ixs[:n_neigh]

        if rank_method is None:
            new_m[i, ixs] = r[ixs]

        elif rank_method is 'binary':
            new_m[i, ixs] = 1

        else:
            new_m[i, ixs] = rankdata(r[ixs], method=rank_method)/len(ixs)

    if axis == 'col':
        new_m = new_m.T

    return new_m

def norm(m, axis='col'):
    if axis == 'col':
        return np.nan_to_num(m / m.sum(0))
    elif axis == 'row':
        return np.nan_to_num(m / m.sum(1)[:,None])

def prune_adj_matrix(adj_file, pruning_method='union', p_neigh = 0.05, clip = (None,None), batch= 1000, atom = tables.Float64Atom(), clear_and_overwrite=True):

    with tables.open_file(adj_file, 'a') as h5:
        dts_in_h5 = set([x.name for x in list(h5.root)])

        rows = np.array(h5.root.rows, dtype=str)
        cols = np.array(h5.root.cols, dtype=str)
        mshape = h5.root.m.shape

        #1) Pruning Rows
        chunkshape = (batch, len(cols))
        n_neigh = int(np.clip(round(len(cols)*p_neigh), clip[0], clip[1]))

        if 'row_prun_ixs' in dts_in_h5:
            h5.root.row_prun_ixs._f_remove()
        out_m = h5.create_carray(h5.root, 'row_prun_ixs', tables.IntAtom(), mshape, chunkshape=chunkshape)

        for i_batch in tqdm(np.arange(0,len(rows), batch), desc='prun. rows', leave=False):
            out_m[i_batch:i_batch+batch, :] = prune(h5.root.m[i_batch:i_batch+batch, :], n_neigh = n_neigh, axis='row', rank_method='binary')

        #2) Pruning Cols
        chunkshape = (len(rows), batch)
        n_neigh = int(np.clip(round(len(rows)*p_neigh), clip[0], clip[1]))

        if 'col_prun_ixs' in dts_in_h5:
            h5.root.col_prun_ixs._f_remove()
        out_m = h5.create_carray(h5.root, 'col_prun_ixs', tables.IntAtom(), mshape, chunkshape=chunkshape)

        for i_batch in tqdm(np.arange(0,len(cols), batch), desc='prun. cols', leave=False):
            out_m[:, i_batch:i_batch+batch] = prune(h5.root.m[:, i_batch:i_batch+batch], n_neigh = n_neigh, axis='col', rank_method='binary')

        #3) Apply pruning method
        if 'm_prun' in dts_in_h5:
            h5.root.m_prun._f_remove()
        out_m = h5.create_carray(h5.root, 'm_prun', atom, mshape, chunkshape=(10000,10000))

        if pruning_method == 'union': #Conservative pruning where rows and cols keep their own edges regardless of the other (i.e. if prune by rows but kept by cols, is kept)
            for rix in tqdm(np.arange(0,len(rows), 10000), desc='merging (method-->union)', leave=False):
                for cix in np.arange(0,len(cols), 10000):
                    out_m[rix:rix+10000, cix:cix+10000] = h5.root.m[rix:rix+10000, cix:cix+10000]*(h5.root.row_prun_ixs[rix:rix+10000, cix:cix+10000]+h5.root.col_prun_ixs[rix:rix+10000, cix:cix+10000]).astype(bool)
        elif pruning_method == 'intersection':
            for rix in tqdm(np.arange(0,len(rows), 10000), desc='merging (method-->intersection)', leave=False):
                for cix in np.arange(0,len(cols), 10000):
                    out_m[rix:rix+10000, cix:cix+10000] = h5.root.m[rix:rix+10000, cix:cix+10000]*(h5.root.row_prun_ixs[rix:rix+10000, cix:cix+10000]*h5.root.col_prun_ixs[rix:rix+10000, cix:cix+10000]).astype(bool)

        else:
            sys.exit('\nThe pruning method "%s" is not implemented! You should program it ;)\n\n'%pruning_method)

    if clear_and_overwrite:
        #Renaming, updating and removing added data
        with h5py.File(adj_file, 'a') as h5:
            #--Removing old data
            for file in ['row_prun_ixs', 'col_prun_ixs']:
                if file in h5.keys():
                    del h5[file]
            #--Updating m
            del h5['m']
            h5['m'] = h5['m_prun']
            del h5['m_prun']

def norm_and_prune_adj_matrix(adj_file, rank_method='min', p_neigh = 0.01, is_undirected=False, batch= 1000, atom = tables.Float64Atom(),
                             clear_and_overwrite=True):

   with tables.open_file(adj_file, 'a') as h5:
        dts_in_h5 = set([x.name for x in list(h5.root)])

        rows = np.array(h5.root.rows, dtype=str)
        cols = np.array(h5.root.cols, dtype=str)
        mshape = h5.root.m.shape

        #1) Preparing m for n1 (m4n1)
        #print_info('Working on m4n1...\n')

        #--Norm
        chunkshape = (len(rows), batch)
        if 'm_col_norm' in dts_in_h5:
            h5.root.m_col_norm._f_remove()

        out_m = h5.create_carray(h5.root, 'm_col_norm', atom, mshape, chunkshape=chunkshape)
        for i_batch in tqdm(np.arange(0,len(cols), batch), desc='norm. cols', leave=False):
            out_m[:, i_batch:i_batch+batch] = norm(h5.root.m[:, i_batch:i_batch+batch], axis='col')

        #--Prune
        chunkshape = (batch, len(cols))
        n_neigh = int(np.clip(round(len(cols)*p_neigh), 1, None))

        # !! If is_undirected then this will create the final m_norm !!
        if is_undirected and len(rows)==len(cols) and np.all(rows==cols):
            if 'm_norm' in dts_in_h5:
                h5.root.m_norm._f_remove()

            out_m = h5.create_carray(h5.root, 'm_norm', atom, mshape, chunkshape=chunkshape)
            for i_batch in tqdm(np.arange(0,len(rows), batch), desc='prun. rows', leave=False):
                out_m[i_batch:i_batch+batch, :] = prune(h5.root.m_col_norm[i_batch:i_batch+batch, :], n_neigh = n_neigh, axis='row', rank_method=rank_method)

        # !! If not is_undirected then we have to compute both directions and merge them !!
        else:
            if 'm4n1' in dts_in_h5:
                h5.root.m4n1._f_remove()

            out_m = h5.create_carray(h5.root, 'm4n1', atom, mshape, chunkshape=chunkshape)
            for i_batch in tqdm(np.arange(0,len(rows), batch), desc='prun. rows', leave=False):
                out_m[i_batch:i_batch+batch, :] = prune(h5.root.m_col_norm[i_batch:i_batch+batch, :], n_neigh = n_neigh, axis='row', rank_method=rank_method)

            #2) Preparing m for n2 (m4n2)
            #print_info('Working on m4n2...\n')

            #--Norm
            chunkshape = (batch, len(cols))
            if 'm_row_norm' in dts_in_h5:
                h5.root.m_row_norm._f_remove()

            out_m = h5.create_carray(h5.root, 'm_row_norm', atom, mshape, chunkshape=chunkshape)
            for i_batch in tqdm(np.arange(0, len(rows), batch), desc='norm. rows', leave=False):
                out_m[i_batch:i_batch+batch, :] = norm(h5.root.m[i_batch:i_batch+batch, :] , axis='row')

            #--Prune
            chunkshape = (len(rows), batch)
            n_neigh = int(np.clip(round(len(rows)*p_neigh), 1, None))

            if 'm4n2' in dts_in_h5:
                h5.root.m4n2._f_remove()

            out_m = h5.create_carray(h5.root, 'm4n2', atom, mshape, chunkshape=chunkshape)
            for i_batch in tqdm(np.arange(0,len(cols), batch), desc='prun. cols', leave=False):
                out_m[:, i_batch:i_batch+batch] = prune(h5.root.m_row_norm[:, i_batch:i_batch+batch], n_neigh = n_neigh, axis='col', rank_method=rank_method)

            #3) Summing both matrices
            #sys.stderr.write('Merging...\n')
            if 'm_norm' in dts_in_h5:
                h5.root.m_norm._f_remove()

            out_m = h5.create_carray(h5.root, 'm_norm', atom, mshape, chunkshape=(10000,10000))
            for rix in tqdm(np.arange(0,len(rows), 10000), desc='merging', leave=False):
                for cix in np.arange(0,len(cols), 10000):
                    out_m[rix:rix+10000, cix:cix+10000] = h5.root.m4n1[rix:rix+10000, cix:cix+10000] + h5.root.m4n2[rix:rix+10000, cix:cix+10000]

   if clear_and_overwrite:
       #Renaming, updating and removing added data
       with h5py.File(adj_file, 'a') as h5:
           #--Removing old data
           for file in ['m_col_norm', 'm_row_norm', 'm4n1', 'm4n2']:
               if file in h5.keys():
                   del h5[file]
           #--Updating m
           del h5['m']
           h5['m'] = h5['m_norm']
           del h5['m_norm']

#-----------------------
# Matrix mutliplication
#-----------------------
def mat_mult_in_chunks(m1_h5_path, m2_h5_path, out_file_path,
                       batch_size=5000, atom=tables.Float64Atom(), dt_name='m'):

    with tables.open_file(out_file_path, 'w') as h5_out: #Opening output file
        with h5py.File(m1_h5_path,'r') as m1: #Operning matrix1
            m1_shape = m1['m'].shape
            with h5py.File(m2_h5_path,'r') as m2:  #Operning matrix2
                m2_shape = m2['m'].shape

                shape = (m1_shape[0], m2_shape[1])

                #Creating output dataset
                out_m = h5_out.create_carray(h5_out.root, dt_name, atom, shape,
                                             chunkshape=(batch_size, batch_size))

                for i in tqdm(range(0, m1_shape[0], batch_size),desc='row_batch', leave=False): #Rows matrix 1
                    for j in tqdm(range(0, m2_shape[1], batch_size), leave=False, desc='column_batch'): # Cols matrix 2
                        for k in range(0, m1_shape[1], batch_size): #Cols matrix 1 | Rows matrix 2

                            out_m[i:i+batch_size, j:j + batch_size] += np.dot(m1['m'][i:i + batch_size, k:k + batch_size],
                                                                          m2['m'][k:k + batch_size, j:j + batch_size])

                h5_out.create_array(h5_out.root,'rows',obj=m1['rows'][:])
                h5_out.create_array(h5_out.root,'cols',obj=m2['cols'][:])

def run_matrix_multiplication(adj_folder, order, prune=False, norm_and_prune=False, p_neigh=0.05, clip=(None, None), pruning_method='union',remove_intra_selfloops=False):

    """
    adj_folder -- folder path with the matrices to be multiplicate. Must be a h5 file named m{}.h5, where {} is a number specificing the dataset position in the metapath.
    order -- the order to multiply the matrices. Must be a list of tuples
    prune -- If true it prunes the network keeping as maximum the proportion specified in 'p_neigh'
    norm_and_prun -- Weights are normalized (by the sum) before prunning
    p_neigh -- Proportion of maximum neighbours to keep if pruning
    clip -- tuple indicating the minimum and maximum number of neighbours to keepduring the pruning. Useful for very small or large neighbours where the 0.05 can be too few/many
    pruning_method -- so far it only accepts 'union' which is a conservative pruning where rows and cols keep their own edges regardless of the other (i.e. if prune by rows but kept by cols, is kept)
    remove_intra_selfloops -- if False (default) it allows having self-loops during the matrix multiplication process (i.e. CPD-int-GEN-int-CPD-trt-DIS will connect CPD to itself as they will share targets by definition). When all the matrix multiplication process is complete, the seelf-loops in final resulting network are removed anyhow, in other words, this only affects to the self-loops appearing during the process but not in the final result.
    """

    if len(order) == 0:
        sys.exit('No order detected!\n')

    for i1, i2 in order:
        m1 = adj_folder+'/m%s.h5'%i1
        m2 = adj_folder+'/m%s.h5'%i2
        new_name = i1+i2

        opath = adj_folder+'/m%s.h5'%("".join(dict.fromkeys(i1+i2))) #This "".join(dict.fromkeys()) is to remove number repetitions (i.e. 12 & 23 --> 123 (and not 1223)

        #--Matrix mult
        mat_mult_in_chunks(m1, m2, out_file_path= opath)

        #--Remove self-loops (-->default is False so this is not run!!)
        if remove_intra_selfloops:
            update_self_loops(opath, value=0)

        if norm_and_prune:
            #--Norm and prune
            """After removing self-loops, it can happen that a node has all zeros. This will raise a "zero division" warning during the
               normalization. Don't worry about it, this node will be removed later on. Remember that we CAN NOT filter out those nodes
               as that would change the shape of the matrix (precalculated before) which would make the matrix multiplication fail."""
            norm_and_prune_adj_matrix(opath,
                                      p_neigh = p_neigh,
                                      is_undirected=False) #The result of a matrix mult is never considered undirected
        elif prune:
            prune_adj_matrix(opath,
            p_neigh = p_neigh,
            clip = clip,
            pruning_method = pruning_method)

    #--Remove self-loops (if any)
    update_self_loops(opath, value=0)

    #--Removing zero lines from the final matrix
    remove_empty_rows_and_cols(opath)

#----------------------
# Aggregating matrices
#----------------------
def get_continuous_portions_in_sorted_array(v):
    portions = []
    last = 0
    for i in range(len(v)-1):
        if v[i+1]-v[i] > 1:
            portions.append((last,i))
            last =i+1
    portions.append((last, len(v)-1))
    return portions

def aggregate_adj_matrices(adj_files, ofile, method='union', batch=1000, atom=tables.Float64Atom()):
    from bisect import bisect_left

    # 1) Getting common universes
    r, c = set([]), set([])
    for adj in adj_files:
        with h5py.File(adj,'r') as f:
            _r = set(f['rows'][:].astype(str))
            _c = set(f['cols'][:].astype(str))

        #--Merge edges at metaedge level
        if method == 'union':
            r = set.union(r, _r)
            c = set.union(c, _c)
        elif method == 'intersection':
            if len(_r) == 0:
                r = _r
                c = _c
            else:
                r = set.intersection(r, _r)
                c = set.intersection(c, _c)
        else:
            sys.exit('Invalid method "%s". Supported methods are: "union", "intersection".\n'%method)

    rows = np.asarray(sorted(r))
    cols = np.asarray(sorted(c))

    # 2) Iterating though each adj file and update final matrix
    mshape = (len(rows), len(cols))
    chunkshape = (batch, mshape[1])

    with tables.open_file(ofile, 'w') as h5_out:
        out_m = h5_out.create_carray(h5_out.root, 'm', atom, mshape, chunkshape=chunkshape)

        if method =='union':
            for adj in adj_files:
                _f = h5py.File(adj,'r')
                _m = _f['m']
                _cols = [bisect_left(cols, c) for c in _f['cols'][:].astype(str)]
                col_portions = get_continuous_portions_in_sorted_array(_cols)

                for i_batch in np.arange(0, _m.shape[0], batch):
                    v = _m[i_batch:i_batch+batch]
                    _rows = [bisect_left(rows, r) for r in _f['rows'][i_batch:i_batch+batch].astype(str)]
                    row_portions = get_continuous_portions_in_sorted_array(_rows)

                    #--adding by windows of continuous values (better performance)
                    for rw in row_portions:
                        for cw in col_portions:
                            cix = _cols[cw[0]:cw[1]]

                            #--aggregating
                            out_m[_rows[rw[0]]:_rows[rw[1]]+1,
                                  _cols[cw[0]]:_cols[cw[1]]+1] += v[rw[0]:rw[1]+1,
                                                                 cw[0]:cw[1]+1]
        elif method == 'intersection':
            sys.exit('The intersection method is not avaiable yet. You have to program it!')

        h5_out.create_array(h5_out.root,'rows',obj=rows.astype('S'))
        h5_out.create_array(h5_out.root,'cols',obj=cols.astype('S'))

#--------
# Stats
#--------
def get_stats_from_adj(adj_file, chunk=1000):

    with h5py.File(adj_file,'r') as f:

        nrows, ncols = f['m'].shape
        nedges = 0
        for i in tqdm(np.arange(0,f['m'].shape[0], chunk), leave=False):
            nedges += (f['m'][i:i+chunk,:].astype(bool)).sum().sum()

    return nrows, ncols, nedges

def get_stats_from_edge_file(file, flip_file=False, method='conservative', skip_header=True):

    if type(file) == str:
        r,c,e = set([]),set([]),0
        for x in gpr.read_edges(file, skip_header=skip_header):
            r.add(x[0])
            c.add(x[1])
            e+=1
        if flip_file:
            c,r = len(r), len(c)
        else:
            r,c = len(r), len(c)

    else:
        if method =='conservative':
            r, c, e = set([]), set([]), set([])
            for i in range(len(file)):
                f = file[i]
                order = [0,1]
                if  flip_file[i]:
                    order = [1,0]
                for x in gpr.read_edges(f, skip_header=skip_header):
                    r.add(x[order[0]])
                    c.add(x[order[1]])
                    e.add((x[0],x[1]))
            r,c,e = len(r), len(c), len(e)

    return r,c,e

def get_dataset_pruning_stats(medges, dts_paths, dts_flip, adj_folder):

    a = ''

    #Getting original file numbers
    for i, medge in enumerate(medges):
        a+='%s\n'%medge
        nds = [medge[:3], medge[-3:]]

        #--pre pruning
        file = dts_paths[i]
        flip = dts_flip[i]
        if len(file) == 1:
            file = file[0]
            flip = flip[0]
        r1,c1,e1 = get_stats_from_edge_file(file, flip, method='conservative', skip_header=True)
        #--after pruning
        r2,c2,e2 = get_stats_from_adj(adj_folder+'/m%i.h5'%(i+1))

        a+='\t-%s:%i-->%i\n'%(nds[0], r1, r2)
        a+='\t-%s:%i-->%i\n'%(nds[1], c1, c2)
        a+='\t-%s:%i-->%i\n'%('EDGES', e1, e2)

    return a

def get_matrix_mult_stats(medges, order, adj_folder):
    a = ""

    s = medges[0]
    for i1, i2 in order:

        s1 = medges[int(i1[0])-1]
        for i in i1[1:]:
            s1+= medges[int(i)-1][3:]

        s2 = medges[int(i2[0])-1]
        for i in i2[1:]:
            s2+= medges[int(i)-1][3:]
        a+='%s x %s\n'%(s1,s2)
        nds = [s1[:3], s2[-3:]]
        file = adj_folder+'/m%s.h5'%(i1+i2)
        r,c,e = get_stats_from_adj(file)
        a+='\t-%s:%i\n'%(nds[0], r)
        a+='\t-%s:%i\n'%(nds[1], c)
        a+='\t-%s:%i\n'%('EDGES', e)
    return a

def write_stats(dt_stats, medges, adj_folder, mxm_order, ofile, action='w'):

    stats_str = ""
    stats_str +='>Metapath\n\t%s\n'%medges2mpath(medges)

    #--metaedge pruning
    stats_str +='\n\t>Metaedge_pruning\n'

    for l in dt_stats.splitlines():
        stats_str+='\t\t%s\n'%l

    #--matrix multiplication
    if len(medges) > 1:
        stats_str +='\n\t>Matrix_multiplication\n'
        mm_stats = get_matrix_mult_stats(medges, mxm_order, adj_folder)
        for l in mm_stats.splitlines():
            stats_str+='\t\t%s\n'%l

    with open(ofile, action) as o:
        if action == 'a':
            o.write('\n')
        o.write(stats_str)
