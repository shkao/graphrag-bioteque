root_path = '../../../'
metadata_path = root_path+'/metadata/'
code_path = root_path+'code/embeddings/'
graph_edges_path = root_path+'/graph/processed/propagated/'

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


#--------
# Utils
#--------

#General
def get_node2abbr(reverse=False):
    d = {
    'Cell':'CLL',
    'Chemical_entity':'CHE',
    'Cellular_component':'CMP',
    'Compound':'CPD',
    'Disease':'DIS',
    'Domain':'DOM',
    'Gene':'GEN',
    'Molecular_function':'MFN',
    'Pathway':'PWY',
    'Perturbagen':'PGN',
    'Pharmacologic_class':'PHC',
    'Tissue':'TIS'
    }
    if reverse:
        return {y:x for x,y in d.items()}
    else:
        return d

def node2abbr(node,reverse=False):
    d = {x.lower():y for x,y in get_node2abbr(reverse=reverse).items()}
    if type(node)==str:
        node = '_'.join(node.split(' ')).lower()
        if node in d:
            return d[node]
        else:
            return None
    else:
        r = []
        for n in node:
            n = '_'.join(n.split(' ')).lower()
            if n in d:
                r.append(d[n])
            else:
                r.append(None)
        return r

def get_undirected_megs(path=metadata_path+'/undirected_metaedges.txt'):
    with open(path,'r') as f:
        return set(f.read().splitlines())

def build_directory_system(p, n_mpaths=1):

    #--Stats
    stats_path = p+'/stats/'
    if not os.path.isdir(stats_path):
        os.mkdir(stats_path)

    #--Nodes
    if not os.path.exists(p+'/nodes'):
        os.mkdir(p+'/nodes')

    #--Scratch
    _scratch_path = p+'/scratch/'
    if not os.path.isdir(_scratch_path):
        os.mkdir(_scratch_path)
    scratch_path= _scratch_path+'/edges/'
    if not os.path.isdir(scratch_path):
        os.mkdir(scratch_path)

    for i in range(n_mpaths):
        i = i+1
        if not os.path.isdir(scratch_path+'/mp%i'%i):
            os.mkdir(scratch_path+'/mp%i'%i)
        #--adj_path
        adj_path = scratch_path+'/mp%i/adj'%i
        if not os.path.isdir(adj_path):
            os.mkdir(adj_path)

def str2bool(s):
    _s = str(s)
    if _s.lower() in set(['t','true']) or _s == '1':
        return True
    elif _s.lower() in set(['f','false', 'none']) or _s == '0':
        return False
    else:
        sys.exit('Unknown bool value: %s'%s)

def parse_user_mxm_order(_order, _medges):
    parsed_mxm = []
    if len(_order) < len(_medges):
            sys.exit('Detected %i metapaths but only %i have matrix multiplication order (mxm_order)\n'%(len(_medges), len(_order)))

    for i in range(len(_order)):
        mxm = _order[i]
        medges = _medges[i]

        if len(medges) == 1:
            parsed_mxm.append([])

        else:
            starts_with_zero= '0' in mxm
            if starts_with_zero:
                mn,mx = 0, len(medges)
            else:
                mn,mx = 1, len(medges)+1

            _d = set([x for x in mxm]).difference(set(['-',','] + list(map(str,np.arange(mn,mx)))))

            if len(_d) > 0:
                sys.exit('Invalid values found in mxm. Allowed values are "-", "," and numbers from %i-%i. Found: "%s"\n'%((mn,mx-1,' '.join(["%s"%x for x in _d]))))

            mxm = mxm.split(',')
            if len(mxm) != len(medges)-1:
                sys.exit('The number of mxm was %i but expected %i for the metapath: %s\n'%(len(mxm), medges2mpath(medges)))

            r = []
            for x in mxm:
                x = x.split('-')
                if int(x[1][0]) - int(x[0][-1]) != 1:
                    sys.exit('Detected no continous multiplication --> %s-%s\n'%(x[0], x[1]))

                if starts_with_zero:
                    x = [''.join([str(int(l)+1) for l in y]) for y in x]
                r.append(tuple(x))

            if set([int(x) for x in r[-1][0] + r[-1][1]]) != set(np.arange(len(medges))+1):
                sys.exit('The final multiplication (%sx%s) do not cover all the multiplications!\n'%(r[-1][0], r[-1][1]))
            parsed_mxm.append(r)

    return parsed_mxm

def autodetect_mxm_order(medges):

    #1 ) Detect if there are more than 2 redundant repetitions (they may complicate order detection)
    for i in range(len(medges)-2):
        k = set.intersection(*[set([x, flip_medge(x)]) for x in medges[i:i+3]])
        if len(k) > 0:
            sys.exit('Detected more than 2 medge repetitions in the metapath: "%s".\nThis may cause inconsistencies during matrix multiplication. Please, redefine the metapath or manually add a mxm order.\nExiting...\n'%medges2mpath(medges[i:i+3]))

    order = []
    if len(medges) >= 2:
        #--Get Priority multiplications (i.e. those that are symmetric)
        _o = []
        for s in range(len(medges)-1):
            for i in range(len(medges)-1):
                if i+s+2 > len(medges): continue

                k = '_'.join(medges[i:i+s+2])

                if k == flip_medge(k):
                    lb = np.arange(i+1,i+s+2+1)
                    lb = '-'.join([str(x) for x in lb])

                    for x in _o:
                        y = ''.join(x.split('-'))
                        if y in lb: continue
                        if x in lb:
                            lb = y.join(lb.split(x))
                        else:
                            for xx in x.split('-'):
                                if y in lb: continue
                                if xx in lb:
                                    lb = lb.replace(xx, ''.join(x.split('-')))
                    _o.append(lb)

        for x in _o:
            x = x.split('-')
            for i in range(len(x)-1):
                y = (''.join(''.join(x[:i+1]).split('-')), x[i+1])
                order.append(y)

        #--Checking covered mxm and adding the remainding
        covered = set([int(i) for x in order for y in x for i in y])
        ixs = np.arange(len(medges))+1

        #-------Messy code to find out the remaining matrix mult
        a = ''
        z = []

        for x in np.arange(len(medges))+1:
            if x in covered:
                a+=str(x)
            else:
                if a != '':
                    z.append(a)
                    a = ''
        if a != '':
            z.append(a)

        if len(z) == 0 or z[0] == '':
            z = [str(len(medges))]

        for ix in range(len(z)):
            if ix == 0:
                init = 0
            else:
                init = int(z[ix-1][-1])-1

            for i in np.arange(init, int(z[ix][0])-1):
                i +=1

                if str(i+1) == z[ix][0]:
                    order.append((''.join(map(str,ixs[:i])), z[ix]))
                    break
                else:
                    order.append((''.join(map(str,ixs[:i])), str(i+1)))
        for i in np.arange(int(z[-1][-1]), len(medges)):
            order.append((''.join(map(str,ixs[:i])), str(i+1)))

    return order



def parse_parameter_file(file, _reference_dataset_path=graph_edges_path):

    #Undirected medges
    _undirected_megs =  get_undirected_megs()

    args = dict(mpaths=None,
                source = '',
                target = '',
                medges=None,
                medges_undirected = None,
                dts_paths=None,
                dts_flip=False,
                dts_w=False,
                mxm_order = None,
                compute_dwpc = True,
                damping_weight = 0.5,
                pruning_cutoffs = (0.05, 3, 250), # (Proportion of total neighbours, min_neighbours, max_neighbours)
                pruning_method = 'union',
                remove_intra_selfloops = False, # Whether to remove self-loops appearing in intermediate steps of the matrix multiplication (i.e. CPD-int-GEN-int-CPD-trt-DIS)
                min_covered_nodes_in_netw = 0.05, # Minimum proportion of node universe required to embed a network component,
                tag_source = None,
                tag_target = None,
                fdirect=False,
                verbose=False,

                #--Specific for precomputed edges
                precomputed_edges=False,
                input_edge_file_sep = 'auto',
                input_edge_file_has_header = True,
                input_edge_file_is_weighted = False,

               )

    #1) Read arguments/parameters
    with open(file,'r') as f:
        for l in f:
            h = [x.strip() for x in l.rstrip('\n').split('\t')]

            #Metapath
            if h[0] == 'metapath':
                METAPATHS = []
                for metapath in h[1:]:
                    if '-' in metapath:
                        metapath = metapath.split('-')
                    elif ',' in metapath:
                        metapath = metapath.split(',')
                    elif '\t' in metapath:
                        metapath = metapath.split('\t')
                    METAPATHS.append(metapath)
                args['mpaths'] = METAPATHS

            #Input data
            elif h[0] == 'datasets':
                DATASETS = []
                for dt in h[1:]:
                    datasets = []
                    for x in dt.split('||'):
                        datasets.append(re.split(',|\+',x))
                    DATASETS.append(datasets)
                args['dts_paths'] = DATASETS

            #Reverse datasets
            elif h[0] in set(['flip_datasets', 'flip_edges', 'reverse_datasets','reverse_edges']):
                args['dts_flip'] = h[1:] #This will be processed later on

            #is_weighted
            elif h[0] == 'is_weighted':
                args['dts_w'] = h[1:] #This will be processed later on

            #compute dwpc
            elif h[0] == 'compute_dwpc':
                args['compute_dwpc'] = h[1:]

            #Damping weight
            elif h[0] == 'damping_weight':
                args['damping_weight'] = float(h[1])

            #Pruning cutoff
            elif h[0] == 'pruning_cutoffs':
                pruning_cutoffs = h[1].split(',')
                args['pruning_cutoffs'] = (float(pruning_cutoffs[0]), int(pruning_cutoffs[1]), int(pruning_cutoffs[2]))

            elif h[0] == 'pruning_method':
                args['pruning_method'] = h[1]

            elif h[0] == 'remove_intra_selfloops':
                args['remove_intra_selfloops'] = str2bool(h[1])

            #Minimum nodes to cover if more than one connected component
            elif h[0] == 'min_covered_nodes_in_netw':
                args['min_covered_nodes_in_netw'] = float(h[1])

            #Matrix multiplication order
            elif h[0] == 'mxm_order':
                args['mxm_order'] = h[1:] #This will be processed later on

            #Node tags

            #--source
            elif h[0] == 'tag_source' or h[0] == 'source_tag':
                while not h[1].endswith('__'):
                    h[1] = h[1]+'_'
                while not h[1].startswith('__'):
                    h[1] = '_'+h[1]
                args['tag_source'] = h[1]

            #--target
            elif h[0] == 'tag_target' or h[0] == 'target_tag':
                while not h[1].endswith('__'):
                    h[1]+='_'
                while not h[1].startswith('__'):
                    h[1] = '_'+h[1]
                args['tag_target'] = h[1]

            #Force_directed flag
            elif h[0] == 'force_directed':
                args['fdirect'] = str2bool(h[1])

            #Verbose flag
            elif h[0] == 'verbose':
                args['verbose'] = str2bool(h[1])

            #----- Precalculated edges arguments------
            elif h[0] == 'precomputed_edges':
                args['precomputed_edges'] = str2bool(h[1])

            elif h[0] == 'input_edge_file_is_weighted':
                args['input_edge_file_is_weighted'] = str2bool(h[1])

            elif h[0] == 'input_edge_file_sep':
                args['input_edge_file_sep'] = h[1]

            elif h[0] == 'input_edge_file_has_header':
                args['input_edge_file_has_header'] = str2bool(h[1])

    if args['mpaths'] is None:
        sys.exit('No metapath was provided, exiting...\n')
    if args['dts_paths'] is None:
        sys.exit('No datasets were provided, exiting...\n')

    #2) Adding remaining argments (inferred from the previous ones)
    args['medges'] = [mpath2medges(mp) for mp in args['mpaths']]
    args['medges_undirected'] = [[medge in _undirected_megs for medge in medges] for medges in args['medges']]
    #--source
    source = np.unique([mpath[0] for mpath in args['mpaths']])
    if len(source) > 1:
        sys.exit('The provided metapaths must have the same source node. Detected: %s'%','.join(source))
    else:
        args['source'] = source[0]
    #--target
    target = np.unique([mpath[-1] for mpath in args['mpaths']])
    if len(target) > 1:
        sys.exit('The provided metapaths must have the same source node. Detected: %s'%','.join(target))
    else:
        args['target'] = target[0]

    #3) Processing arguments/parameters

    #--Matching flip/weight with dataset length
    for flag in set(['dts_flip', 'dts_w', 'compute_dwpc']) & set(args):

        restructured_flag = []
        q = args[flag]
        if type(q) == str:
            q = str2bool(q)
        elif type(q) == list and len(q)==1 and ('||' not in q[0] and ',' not in q[0]):
            q = str2bool(q[0])

        if type(q) == bool:
            args[flag] = [[[q for z in y] for y in x] for x in args['dts_paths']]
        else:

            for i1,d in enumerate(q):
                _rst_flag = []
                for i2,x in enumerate(d.split('||')):
                    _d = []
                    for boolean in x.split(','):
                        _d.append(str2bool(boolean))

                    if len(_d) <len(DATASETS[i1][i2]):
                        _d = _d*len(DATASETS[i1][i2])
                    _rst_flag.append(_d)
                restructured_flag.append(_rst_flag)
            args[flag] = restructured_flag

    #--Matrix multiplication order (mxm)
    if not args['mxm_order']:
        args['mxm_order'] = []
        for medges in args['medges']:
            args['mxm_order'].append(autodetect_mxm_order(medges))
    else:
        args['mxm_order'] = parse_user_mxm_order(args['mxm_order'], args['medges'])

    #5) Inferring dataset paths if not given
    if not args['precomputed_edges']:
        dataset_paths = []
        for I in range(len(args['medges'])):
            _dataset_paths = []
            for i, medge in enumerate(args['medges'][I]):

                new_path_dt = []
                for dt in args['dts_paths'][I][i]:
                    if not os.path.isfile(dt):
                        if medge in set(os.listdir(_reference_dataset_path)):
                            _medge = medge
                        elif flip_medge(medge) in set(os.listdir(_reference_dataset_path)):
                            _medge = flip_medge(medge)
                        else:
                            sys.exit('Metaedge "%s" is not in the reference dataset path: "%s"'%(medge,_reference_dataset_path))

                        new_path_dt.append(_reference_dataset_path+'/%s/%s'%(_medge,dt))
                    else:
                        new_path_dt.append(dt)
                _dataset_paths.append(new_path_dt)
            dataset_paths.append(_dataset_paths)

        args['dts_paths'] = dataset_paths

    return args

def mpath2medges(metapath):
    """Return all the metaedge (triplets) in a given metapath"""
    if type(metapath) == str or type(metapath) ==np.str_:
        mp = metapath.split('-')
    else:
        mp = metapath

    metaedges = []
    for i in np.arange(0,len(mp)-1,2):
        metaedges.append('-'.join(mp[i:i+3]))
    return metaedges

def medges2mpath(medges):
    """Return the full metapath from a given list of metaedges (triplets)"""
    return '-'.join([medges[0]]+['-'.join(x.split('-')[1:]) for x in medges[1:]])

def mpath2str(mp):
    if type(mp) == list:
        return '-'.join(mp)
    else:
        return str(mp)

def flip_medge(meg, directionality_character='_'):
    """Flips the metaedge direction"""
    if type(meg) == str:
        return '-'.join([directionality_character+x[:-1] if x.endswith(directionality_character)
                         else x[1:]+directionality_character if x.startswith(directionality_character)
                         else x for x in  meg.split('-')[::-1]])
    else:
        return [directionality_character+x[:-1] if x.endswith(directionality_character)
                else x[1:]+directionality_character if x.startswith(directionality_character)
                else x for x in  meg[::-1]]

def metapath2forward(metapath):
    if type(metapath) == str:
        mp = metapath.split('-')
    else:
        mp = metapath
    i = int((len(mp)/2) + 0.5 )
    i+= 1-i%2
    return mp[:i]

def traverse_mpaths_from_mpath(metapath):
    """Return all the possible metapath starting for each node in a given metapath"""
    if type(metapath) == str:
        mp = metapath.split('-')
    else:
        mp = metapath
    metapaths = []
    for i in range(0,len(mp)-1,2):
        q = mp[i:] + mp[1:i+1]
        metapaths.append(q)
    return metapaths

def extend_sources_through_mpaths(mpath,sources):
    medges  = mpath2medges(mpath)
    srcs_iv = sources[::-1]
    v = sources + sources[::-1]
    while len(v) < len(medges):
        v+= sources[::-1]
    return v[:len(medges)]
