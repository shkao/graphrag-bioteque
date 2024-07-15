root_path = '../../../'
graph_edges_path = root_path+'/graph/processed/propagated/'
code_path = root_path+'code/embeddings/'
emb_path = root_path+'embeddings/'

import os
import sys
import copy
import numpy as np
import pandas as pd
import h5py
import random
import subprocess
import networkx as nx
from collections import Counter
import time
from tqdm import tqdm
sys.path.insert(0,code_path)
from utils.utils import mpath2medges, flip_medge, metapath2forward

#--- Plot options
import matplotlib.pyplot as plt
import seaborn as sns
#Fontname
fname ='Courier_New'
#FontSize
fsize = 14

#FONT
font = {'family' : fname,
        'size'   : fsize }

plt.rc('font', **font)

plt.rcParams['pdf.fonttype'] = 42
sns.set(font_scale=1.25,font="Courier_New")
sns.set_style("whitegrid")
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
#----------------

def get_ajd_row_col(adj):
    with h5py.File(adj,'r') as f:
        return f['rows'][:].astype(str), f['cols'][:].astype(str)

def read_edges(path, undirected=False, sep='auto', skip_header=False, adjancent_matrix=False):
    if path.endswith('.h5'):
        with h5py.File(path,'r') as f:
            if adjancent_matrix:
                rows = f['rows'][:].astype(str)
                cols = f['cols'][:].astype(str)
                for i in tqdm(range(len(rows)), desc='Reading rows', leave=False):
                    r = f['m'][i]
                    if undirected:
                        r = r[i:]
                        ixs = np.where(r.astype(bool))[0]
                        for j in ixs:
                          yield rows[i], cols[j+i], r[j]
                    else:
                        ixs = np.where(r.astype(bool))[0]
                        for j in ixs:
                            yield rows[i], cols[j], r[j]
            else:
                batch = 10000
                with h5py.File(path, 'r') as f:
                    for batch_ix in np.arange(0,f['edges'].shape[0],batch):
                        egs = f['edges'][batch_ix:batch_ix+batch]
                        if 'weights' in f.keys():
                            ws = f['weights'][batch_ix:batch_ix+batch]
                            for ix in range(len(egs)):
                                yield list(egs[ix]) + [ws[ix]]
                        else:
                            for e in egs:
                                yield list(e)
    else:
        if sep =='auto' or sep is None:
            if path.endswith('.tsv'):
                sep = '\t'
            elif path.endswith('.csv'):
                sep = ','
            else:
                sys.exit('The path :"%s" has not a known suffix (.h5, .tsv, .csv). You must provide a separator!\n'%path)
        with open(path,'r') as f:
            if skip_header:
                f.readline()

            for l in f:
                if l.startswith('n1\tn2'): continue # auto skip header
                if undirected:
                    h = l.rstrip('\n').split(sep)
                    yield sorted(h[:2]) + h[2:]
                else:
                    yield l.rstrip('\n').split(sep)

def get_graph_from_edge_list(path, source_label=None, target_label=None, weighted=False): #Used for getting stats

    gx = nx.Graph()

    EDGES = set([])
    NODES = {}
    if type(source_label) == np.str_ or type(source_label) == str:
        NODES[source_label] = set([])
    else:
        for label in set(source_label):
            NODES[label] = set([])
    if type(target_label) == np.str_ or type(target_label) == str:
        NODES[target_label] = set([])
    else:
        for label in set(target_label):
            NODES[label] = set([])

    if type(path) == str:
        it = read_edges(path)
    else:
        it = (r for r in m)

    for ix,e in enumerate(it):
        if type(source_label) == np.str_ or type(source_label) == str:
            NODES[source_label].add(e[0])
        else:
            NODES[source_label[ix]].add(e[0])

        if type(target_label) == np.str_ or type(target_label) == str:
            NODES[target_label].add(e[1])
        else:
            NODES[target_label[ix]].add(e[1])

        if weighted:
            EDGES.add(tuple(e[:3]))
        else:
            EDGES.add(tuple(e[:2]))

    #Creating the graph
    for lb,nodes in NODES.items():
        gx.add_nodes_from(list(nodes), label=lb)
    if weighted:
        gx.add_weighted_edges_from(list(EDGES))
    else:
        gx.add_edges_from(list(EDGES))

    return gx

def read_node2neigh(edges, is_weighted=False, map_nd=None):
    """
    Given a list of edges or a path it returns dictionary with each node and its neighbors.
    If a 3rd column is detected it assumes that the edges are weighted so it returns a 2nd dictionary with the weights
    """
    n_to_neigh = {}
    n_to_w = {}

    #1) Reading neighboorhod and weights
    if type(edges) == str:
        edges = read_edges(edges)

    for h in edges:

        if map_nd:
            if type(map_nd) == dict:
                h[:2] = [map_nd[x] for x in h[:2]]
            else:
                h[:2] = map(map_nd,h[:2])

        if len(h) >2:
            is_weighted = True

        #--source node (n1)
        if h[0] not in n_to_neigh:
            n_to_neigh[h[0]] = []
            n_to_w[h[0]] = []
        if h[1] not in n_to_neigh:
            n_to_neigh[h[1]] = []
            n_to_w[h[1]] = []

        n_to_neigh[h[0]].append(h[1])
        n_to_neigh[h[1]].append(h[0])

        try:
            n_to_w[h[0]].append(float(h[2]))
            n_to_w[h[1]].append(float(h[2]))
        except IndexError:
            n_to_w[h[0]].append(1)
            n_to_w[h[1]].append(1)

    if is_weighted:
        return n_to_neigh, n_to_w
    else:
        return n_to_neigh

def read_metaedges(datasets,metapath,approach='merged',
               path=graph_edges_path):

    """Provided a metapath and list of datasets it returns a list of edges from the dataset sorted according to the metapaths

    KeyWords

        -approach <flag>
            How to process the repeated edges across datasets. Currently supports 2 flags:
                *merged* -> Repeated edges are removed
                *weighted* -> It returns a third column with the number of datasets reporting the edge
        -path
            The path from where the data is fetched. Notice that by default is set to the propagated reference_data
    """

    r = []

    medges = mpath2medges(metapath)
    if len(medges) != len(datasets):
        sys.exit('The number of datasets (%i) does not match with the number of metaedges in the metapath (%i)'%(len(datasets),len(medges)))

    for i in range(len(medges)):
        medge = medges[i]
        dts = datasets[i]
        sr = []

        if type(dts) == str:
            dts = [dts]

        for dt in dts:
            if not os.path.exists(path+'/%s/%s.tsv'%(medge,dt)):
                medge = flip_medge(medge)
                reverse = True
            else:
                reverse = False

            with open(path+'/%s/%s.tsv'%(medge,dt),'r') as f:
                f.readline()
                for l in f:
                    h = l.rstrip('\n').split('\t')
                    sr.append(tuple(h[:2]))

        #Checking and applying the approach
        if approach == 'merged':
            sr = list(set(sr))
        elif approach == 'weighted':
            sr = [list(x)+[y] for x,y in Counter(r[medge]).items()]

        if reverse:
            sr = [list(reversed(x[:2]))+list(x[2:]) for x in sr]

        r.append(np.asarray(sr))
    return r

def read_metaedge_dict(metaedge2dataset,approach='merged',
               path=graph_edges_path):

    """Provided a dictionary with metapath and list of datasets it returns a dictionary of metapath and list of edges

    KeyWords

        -approach <flag>
            How to process the repeated edges across datasets. Currently supports 2 flags:
                *merged* -> Repeated edges are removed
                *weighted* -> It returns a third column with the number of datasets reporting the edge
        -path
            The path from where the data is fetched. Notice that by default is set to the propagated reference_data
    """
    r = {}
    #Iterating across metaedges
    for metaedge,datasets in metaedge2dataset.items():
        if type(datasets) == str:
            datasets = [datasets]
        r[metaedge] = []
        #Iterating across datasets
        for dataset in datasets:
            with open(path+'/%s/%s.tsv'%(metaedge,dataset),'r') as f:
                f.readline()
                for l in f:
                    h = l.rstrip('\n').split('\t')
                    r[metaedge].append(tuple(h[:2]))

        #Checking and applying the approach
        if approach == 'merged':
            r[metaedge] = list(set(r[metaedge]))
        elif approach == 'weighted':
            r[metaedge] = [tuple(list(x)+[y]) for x,y in Counter(r[metaedge]).items()]
    return r

def combine_metaedges(metaedge_dict,metaedges_to_combine,logic='or',new_label=None):
    """Returns a new metaedge_dict where a set of given metaedges are combined according to the logic

    KeyWords

        -metaedge_dict <dict>
            Dictionary with metaedge as key and a list of edges (tuples) as values
        -metaedges_to_combine
            List of metaedges that should be combined. Must be the available in the metaedge_dict
        -logic <flag>
            *or* -> Union of all the edges
            *and* -> Intersection of all the edges
        -new_label <str>
            Name of the new metaedge label. If none it uses the metanodes of the first metaedge to combine with the edge "new"


    """
    new_dict = copy.deepcopy(metaedge_dict)

    if new_label is None:
        a = metaedges_to_combine[0].split('-')
        new_label = '%s-new-%s'%(a[0],a[2])

    if logic == 'or':
        new_set = set([])
        for me in metaedges_to_combine:
            new_set.update(metaedge_dict[me])
            del new_dict[me]

        new_dict[new_label] = sorted(new_set)

    elif logic == 'and':
        me = metaedges_to_combine[0]
        new_set = set(metaedge_dict[me])
        del new_dict[me]
        for me in metaedges_to_combine[1:]:
            new_set = new_set & set(metaedge_dict[me])

        new_dict[new_label] = sorted(new_set)

    return new_dict

def add_n_shortcuts(edges,n,directed=False):
    """
    Given a list of edges and a depth "n" it returns all the edges you can get from each node to the
    rest of nodes by going n times through their interactions, i.e.,

    Given [(1,2),(2,3)] and n == 2, returns:
    [(1,2),(2,3),(1,3)]

    You should take into account that this not make sense if with heterogeneous edges!
    """
    updated_edges = set(copy.copy(edges))

    #Creating the dictionary of the interactions
    d = {}
    for e in updated_edges:
        if e[0] not in d:
            d[e[0]] = set([])
        if e[1] not in d:
            d[e[1]] = set([])
        d[e[0]].add(e[1])
        if directed is False:
            d[e[1]].add(e[0])

    #Defining the recursion function
    def it(i,n):
        r = set([])

        childs = d[i]

        c = 0
        for j in childs:
            r.add(j)
            for _ in range(n-1):
                c+=1
                r.update(it(j,n-c)) #recursion
        return r

    #For each node, adds the shortcuts of depth n
    for nd in d:
        #It assures that only one direction is provided and not allow self interactions
        updated_edges.update([(nd,x) for x in it(nd,n) if (x,nd) not in updated_edges and x!=nd])

    return updated_edges

def get_stats_from_metaedge_dict(d):
        stats = {}
        for me in d:
            n1,e,n2 = tuple(me.split('-'))
            st = {n1:set([]),n2:set([]),'edges':0}
            for r in d[me]:
                st[n1].add(r[0])
                st[n2].add(r[1])
                st['edges'] +=1
            st[n1] = len(st[n1])
            if n1 != n2:
                st[n2] = len(st[n2])
            stats[me] = st
        return stats

def prune_metaedge_dict(metaedge_dict, metapath, directed_megs=[], verbose=True, write_to=None):
    """
    Given a metaedge_dict and a metapath it prunes it to assure that the metapath will not be stacked in any node.
    By default it assumes that every metaedge in the network is undirected so it considers the reciprocal of homogeneous metaedges.
    """

    #Check metapath
    if type(metapath) == str:
        _metapath = metapath.split('-')
    else:
        import copy
        _metapath = copy.copy(metapath)

    #If the metapath is not symmetric it is transformed to be symmetric --> needed to check the reverse path
    if _metapath != _metapath[::-1]:
        _metapath += _metapath[::-1][1:]

    #Getting a copy of the metaedge_dict
    new_d = {x: np.asarray(metaedge_dict[x]) for x in metaedge_dict}

    # If there are undirected metaedges the reciprocal edges are added to the dictionary
    for me in new_d:
        if me not in directed_megs:
            n1,e,n2 = tuple(me.split('-'))
            if n1 == n2:
                v = new_d[me]
                v = np.append(v,new_d[me][:,::-1],axis=0)
                v = np.unique(v,axis=0)
                new_d[me] = v

    #Spliting the metapath in metaedges
    megs = [me for me in mpath2medges(_metapath)]

    #Iterating across the metaedges
    for i in range(len(megs)-1):

        me1, ix1 = megs[i], 1
        me2, ix2 = megs[i+1], 0

        #If the metaedge is fliped (backward in the metapath), we change the orientation
        if me1 not in new_d and flip_medge(me1) in new_d:
            me1 = flip_medge(me1)
            ix1 = 1-ix1
        if me2 not in new_d and flip_medge(me2) in new_d:
            me2 = flip_medge(me2)
            ix2 = 1-ix2

        #Getting universes of the bridging nodes between metaedges
        v1 = new_d[me1]
        v2 = new_d[me2]
        uv = list(set(v1[:,ix1]) & set(v2[:,ix2]))

        #Removing every edge not bridged to the following metaedge (according to the metapath)
        v1 = v1[np.in1d(v1[:,ix1], uv)]
        v2 = v2[np.in1d(v2[:,ix2], uv)]

        #Updating metaedge dict
        new_d[me1] = v1
        new_d[me2] = v2

    #Removing those reciprocal undirected edges added add the begging, keeping the original provided edges

    for me in new_d:
        if me not in directed_megs:
            n1,e,n2 = tuple(me.split('-'))
            if n1 == n2:
                U = set([tuple(x) for x in new_d[me]])
                new_d[me] = np.asarray([pair for pair in metaedge_dict[me] if pair in U])

    if verbose or write_to:
        before_stats = get_stats_from_metaedge_dict(metaedge_dict)
        after_stats = get_stats_from_metaedge_dict(new_d)

        stats_log = ''
        for me in before_stats:
            n1,e,n2 = tuple(me.split('-'))
            stats_log+='>%s\n'%me
            stats_log+='%s: %i --> %i (%.2f%%)\n'%(n1,before_stats[me][n1],after_stats[me][n1], (after_stats[me][n1]/before_stats[me][n1])*100)
            if n1 != n2:
                stats_log+='%s: %i --> %i (%.2f%%)\n'%(n2,before_stats[me][n2],after_stats[me][n2], (after_stats[me][n2]/before_stats[me][n2])*100)
            stats_log+='Edges: %i --> %i (%.2f%%)\n'%(before_stats[me]['edges'],after_stats[me]['edges'], (after_stats[me]['edges']/before_stats[me]['edges'])*100)

        #--Adding total
        total = {'Edges':0}
        for me in new_d:
            n1,e,n2 = tuple(me.split('-'))
            if n1 not in total:
                total[n1] = 0
            if n2 not in total:
                total[n2]= 0

            if n1 == n2:
                total[n1] += len(set(new_d[me].ravel()))
            else:
                total[n1] += len(set(new_d[me][:,0]))
                total[n2] += len(set(new_d[me][:,1]))
            total['Edges']+= len(new_d[me])

        stats_log+='>TOTAL\n'
        for x in sorted(total):
            stats_log+='%s: %i\n'%(x,total[x])

        #Printing
        if verbose:
            sys.stderr.write('\n\n')
            sys.stderr.write(stats_log)

        #Writing
        if write_to:
            with open(write_to,'w') as o:
                o.write(stats_log)

    return new_d

def adjancent_matrix_from_edge_list(edge_list, undirected=False, rows= None, cols=None):
    """
    Given a edge list it return an adjancent matrix going from the first position (aka. rows) to the second position (aka columns)
    If undirected, it returns a square matrix where rows and columns are the same
    """

    from bisect import bisect_left
    edge_list = np.asarray(list(edge_list))

    if rows is None:
        rows = np.unique(edge_list[:,0])
    if cols is None:
        cols = np.unique(edge_list[:,1])

    if undirected:
        rows = np.unique(np.concatenate((rows,cols)))
        cols = np.unique(np.concatenate((rows,cols)))

    matrix = np.zeros(shape=(len(rows),len(cols)))
    row_ixs = [bisect_left(rows, x) for x in edge_list[:,0]]
    col_ixs = [bisect_left(cols, x) for x in edge_list[:,1]]
    matrix[row_ixs,col_ixs] = 1
    return pd.DataFrame(matrix, index = rows, columns = cols)

def remove_and_bridge_metaedge_dict(metaedge_dict,metapath,metanodes_to_remove):
    """
    Given a metaedge dict and the metapath it removes from the dictionary those metaedges containing metanodes_to_remove.
    Then it creates a new "artificial" metaedge with those nodes that are interacting after the removing.

    Ex:
        >Input
        .........
        metaedge_dict -- {'PGN-bfn-CLL': [[pgn1,cll1]...], 'PGN-dwr-GEN': [[pgn1,gen1]...]}
        metapath -- CLL-bfn-PGN-dwr-GEN-dwr-PGN-bfn-CLL
        metanodes_to_remove -- set([PGN]).

        >Output
        .........
        new_d --> {'CLL-(bfn-PGN-dwr)-GEN': [cll1,gen1]...}

    and a set of metanodes to remove it removes the men
    """

    if type(metanodes_to_remove) == str:
        metanodes_to_remove = set([metanodes_to_remove])
    if type(metapath) == str:
        metapath = metapath.split('-')
    if metapath[0] in metanodes_to_remove:
        sys.exit('Is not possible to remove the source node. You must reformulate the metapath before continuing. Exiting...\n')

    if not metapath[:int(np.ceil(len(metapath)/2))] == metapath[int(len(metapath)/2):][::-1]:
        sys.exit('Metapath must be circular and symmetric! Exiting...\n')

    new_d = copy.copy(metaedge_dict)

    #Calculating the universes
    uvs = {}
    for me in metaedge_dict:
        n1,e,n2 = me.split('-')
        if n1 not in uvs:
            uvs[n1] = set([])
        if n2 not in uvs:
            uvs[n2] = set([])
        uvs[n1].update(metaedge_dict[me][:,0])
        uvs[n2].update(metaedge_dict[me][:,1])
    uvs = {x:np.unique(list(uvs[x])) for x in uvs}

    #Getting metapath to follow
    forward_path = metapath2forward(metapath)
    III = [0]+[ix for ix,x in enumerate(forward_path) if ix%2 ==0 and x not in metanodes_to_remove]

    for i in range(len(III)-1):
        s1 = III[i]
        s2 = III[i+1]

        mp = forward_path[s1:s2+1]
        megs = mpath2medges(mp)
        if len(megs) < 2: continue #This happens when there is no metanode to remove

        matrices = []
        for me in megs:
            n1,e,n2 = me.split('-')
            rows = uvs[n1]
            cols = uvs[n2]
            if me not in metaedge_dict:
                edge_list = metaedge_dict[flip_medge(me)][:,::-1]
            else:
                edge_list = metaedge_dict[me]

            matrices.append(adjancent_matrix_from_edge_list(edge_list, rows=rows, cols=cols))

        #Connecting source node to final target node
        dist_matrix = matrices[0]
        if len(matrices) > 1:
            for i in np.arange(1,len(matrices)):
                dist_matrix = np.dot(dist_matrix,matrices[i])

        #Getting final matrix
        rows = matrices[0].index.values
        cols = matrices[-1].columns
        dist_matrix = pd.DataFrame(dist_matrix, index = rows , columns = cols)

        #Getting new edges from final matrix
        label = mp[0]+'-('+'_'.join(mp[1:-1])+')-'+mp[-1]
        new_e = []
        for n1 in rows:
            v = cols[np.where(dist_matrix.loc[n1]==1)]
            new_e += list(set(zip([n1]*len(v),v)))
        new_d[label] = np.asarray(new_e)

        #Removing old metaedges
        for me in megs:
            if me not in new_d:
                me = flip_medge(me)
            del new_d[me]

    return new_d

def write_node_input(edges,output,n=100):
    """
    It splits the node universe found in the whole metaedge_dict leaving n nodes (comma separated) per line.

    --edges: Can be a list/set of tuples or a metaedge_dict. If a metaedge_dict is provided it considers only the values of the dictionary as edges.
    """

    uv = set([])
    if type(edges) == dict:
        e = []
        for me in edges:
            e+=list(edges[me])
        edges = e
    for r in edges:
        uv.update(r)

    uv = sorted(uv)

    with open(output,'w') as o:
        for i in range(0,len(uv),n):
            stack = uv[i:i+n]
            o.write(','.join(map(str,stack))+'\n')

def gini_coef(array):
    """Given an array it returns the Gini coefficient"""

    if type(array) != np.ndarray:
        array = np.asarray(array)
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.min(array) <0:
        sys.exit('The input array must be non-negative')
    array = array + 0.0000001 #values cannot be 0

    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1)
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def get_neg_edges_from_egs_simple(edges, is_undirected=False, allow_self_loops=False, not_allowed_edges={}, disable_tqdm=False):

    """
    Given a set of edges it returns the same number of negative edges.

    Keywords
    ---------
    - egs: List of edges
    - is_undirected: If True it assumes that the edges are undirected. If False it only return False edges from the first column to the second.
    - allow_self_loops: If True it allows self-loops as "False edges" (not recommended for LP exercises)
    - not_allowed_edges: dictionary mapping each node to the nodes it can not be paired
    """

    d = {}
    trg_uv = set([])
    for e in edges:
        if e[0] not in d:
            d[e[0]] = set([])
        d[e[0]].add(e[1])
        trg_uv.add(e[1])

        if is_undirected:
            if e[1] not in d:
                d[e[1]] = set([])
            d[e[1]].add(e[0])
            trg_uv.add(e[0])

    neg_edges = []
    for nd in tqdm(d, desc='Getting neg edges', leave=False, disable=disable_tqdm):

        c = len(d[nd])
        uv = trg_uv - d[nd]
        if nd in not_allowed_edges:
            uv = uv - not_allowed_edges[nd]

        if not allow_self_loops:
            uv.discard(nd)
        N = min([len(uv),c])

        neg_edges+=zip([nd]*N,np.random.choice(list(uv),N,replace=False))

    if is_undirected:
        neg_edges = list(set(map(lambda x: tuple(sorted(x)),neg_edges)))

    neg_edges = np.asarray(neg_edges)[np.random.permutation(np.arange(len(neg_edges)))][:len(edges)]
    assert len(set(map(tuple,edges)) & set(map(tuple,neg_edges))) == 0

    return neg_edges


def get_neg_edges_from_egs_conservative(edge_list,desired_edges=None,try_edge_swap=True, max_tries=100,
                                        is_undirected=False, allow_self_loops=False, not_allowed_edges=set([]),
                                        only_edge_swap=False, stop_if_less_edges = False, disable_tqdm=False):

    """
    Given a set of edges it returns a set of N negative edges.
    It tries to keep the same degree distribution of the nodes by doing edge swaping + random subsampling using degrees as probabilities.

    Keywords
    ---------
    - egs: List of edges
    - desired_edges: Number of desired edges. If None, it returns the number of given egs.
    - try_edge_swap: If True it tries to preserve the node degree in the negative edges by double swaping the edges.
                     If not possible it first tries to get random edges for those unswaped nodes.
                     Finally, it gets random negative edges until gettin the number of desired edges.
    - max_tries: Number of double edge swaps tries given for each edge.
    - is_undirected: If True it assumes that the edges are undirected. If False it only return False edges from the first column to the second.
    - allow_self_loops: If True it allows self-loops as "False edges" (not recommended for LP exercises)
    - not_allowed_edges: set of edges you don't want as negative (default: set([]))
    - only_edge_swap: If True the function stops after the edge_swap even if the number of desired edges was not fullfilled (default: False)
    - stop_if_less_edges: If True the function stops if there are less possible edges than desired edges (default: False)

    """

    #1) Setting variables
    egs = np.asarray([r[:2] for r in edge_list]) # Skipping weights
    if desired_edges == None:
        desired_edges = len(egs)
    desired_edges = int(desired_edges)
    remaining_edges = desired_edges
    not_allowed_edges = set(map(tuple,not_allowed_edges))

    edges_without_swap = []
    if is_undirected:
        egs_to_swap = set([tuple(sorted(x)) for x in egs])
        pos_edges = set([tuple(sorted(x)) for x in egs])
        n1s = set([i for x in egs for i in x])
        n2s = n1s
        uvs = set([tuple(sorted(x)) for x in egs])
        not_allowed_edges.update([e[::-1] for e in not_allowed_edges])
        possible_edges = (len(n1s|n2s)**2) - len(pos_edges) - len(not_allowed_edges)
    else:
        egs_to_swap = set([tuple(x) for x in egs])
        pos_edges = set([tuple(x) for x in egs])
        n1s = set([x[0] for x in egs])
        n2s = set([x[1] for x in egs])
        uvs = set([tuple(x) for x in egs])
        possible_edges = (len(n1s)*len(n2s)) - len(pos_edges) - len(not_allowed_edges)

    if desired_edges > possible_edges:
        sys.stderr.write('The number of desired edges (%i) is higher than possible negative edges (%i)\n'%(desired_edges, possible_edges))
        if stop_if_less_edges:
            sys.exit('Exiting...')
        else:
            sys.stderr.write('Number of desired edges was set to %i'%possible_edges)
            desired_edges = possible_edges

    #2) Starting to edge doble swaping
    if try_edge_swap:
        G = nx.Graph()
        G.add_edges_from(egs)

        for nd in tqdm(sorted(G.nodes()), desc='Edge swapping',leave=False, disable=disable_tqdm): #Sorting the nodes is needed when the edges are undirected
            neighboors = set(G[nd])
            for neigh in neighboors:
                if nd in n1s and neigh in n2s:
                    ix = 0
                elif nd in n2s and neigh in n1s:
                    ix = 1
                else:
                    sys.exit('ERROR')
                if is_undirected:
                    e1 = tuple(sorted([nd,neigh]))
                else:
                    e1 = [0,0]
                    e1[ix] = nd
                    e1[abs(1-ix)] = neigh
                    e1 = tuple(e1)

                flag = False
                C = 0
                tried_edges = set([])
                while flag == False :
                    if C == max_tries or len(egs_to_swap)==0:
                        edges_without_swap.append(nd)
                        egs_to_swap.update(tried_edges)
                        tried_edges = set([])
                        break

                    e2 = egs_to_swap.pop() #Here is safe to use pop as the set of tuples is not sorted by id (there are no apparent bias and allows a super fast edge swap)
                    ne1 = (e1[0],e2[1])
                    ne2 = (e2[0],e1[1])

                    if is_undirected:
                        ne1 = tuple(sorted(ne1))
                        ne2 = tuple(sorted(ne1))

                    if allow_self_loops == False and (ne1[0] == ne1[1] or ne2[0] == ne2[1]):
                        tried_edges.add(e2)
                        C +=1
                        continue

                    if ne1 in not_allowed_edges or ne2 in not_allowed_edges:
                        tried_edges.add(e2)
                        C +=1
                        continue

                    if not ne1 in uvs and not ne2 in uvs:
                        uvs.add(ne1)
                        uvs.add(ne2)

                        G.remove_edge(*e1)
                        G.remove_edge(*e2)
                        egs_to_swap.update(tried_edges)
                        egs_to_swap.discard(e1)
                        tried_edges = set([])
                        C = 0
                        break
                    else:
                        tried_edges.add(e2)
                        C +=1

    #3) Getting final negative edges
    neg_edges = uvs.difference(pos_edges)
    lacking_edges = desired_edges - len(neg_edges)
    if only_edge_swap is True:
        if lacking_edges > 0:
            sys.stderr.write("--> %i edges couldn't be generated with edge swap\n"%lacking_edges)
        return neg_edges

    if lacking_edges > 0:

        #--Getting node universes
        _U = set(list(neg_edges)+list(map(tuple,egs)))
        if is_undirected:
            _U = set(map(lambda x: tuple(sorted(x)), _U))

        all_edges = np.asarray(list(_U))

        #--Getting discarted edges per node
        n2e = {} # <-- Edges annotated here will be discarted (i.e. already existing + not allowed)
        for r in all_edges:
            if r[0] not in n2e:
                n2e[r[0]] = set([])
            if r[1] not in n2e:
                n2e[r[1]] = set([])

            n2e[r[0]].add(r[1])
            n2e[r[1]].add(r[0])

            if allow_self_loops is False:
                n2e[r[0]].add(r[0])
                n2e[r[1]].add(r[1])
        #----Adding not_allowed edges to avoid considering them when sampling
        for r in not_allowed_edges:
            if r[0] in n2e:
                n2e[r[0]].add(r[1])
            if r[1] in n2e:
                n2e[r[1]].add(r[0])

        #--Getting node probabilities
        probs = {}
        _neg_edges_count = Counter(np.asarray(list(neg_edges)).ravel())
        _pos_edges_count = Counter(egs.ravel())
        for _n,c in _pos_edges_count.items():
            if _n in _neg_edges_count:
                c = max([0,c- _neg_edges_count[_n]])
            probs[_n] = c

        del _neg_edges_count
        del _pos_edges_count

        #--Getting final list of nodes and probs
        nds = np.unique(list(probs))
        probs = np.asarray([probs[n] for n in nds])

        if is_undirected:
            nds_U = set(nds)
        else:
            nd1_U = set(all_edges[:,0])
            nd2_U = set(all_edges[:,1])

        #Getting remaining negative edges
        pbar = tqdm(total=lacking_edges, leave=False, desc='Getting remaining negative edges', disable=disable_tqdm)
        _count = 0
        while len(neg_edges) < desired_edges:
            _count+=1
            if _count >= 1e8:
                sys.exit('The while loop exceed 1e8 iterations. Please check if the function is running properly...\n')
            pbs = (probs+1e-8)/sum(probs+1e-8)

            _n1 = np.random.choice(nds,p=pbs)

            if is_undirected:
                try:
                    _n2 = random.sample(nds_U.difference(n2e[_n1]),1)[0] #Import NOT to use .pop() as always gets the first element (introducing a bias)
                except ValueError: continue
                _e = tuple(sorted([_n1,_n2]))
            else:
                if _n1 in nd1_U:
                    try:
                        _n2 = random.sample(nd2_U.difference(n2e[_n1]),1)[0]
                    except ValueError: continue
                    _e =(_n1,_n2)
                else:
                    try:
                        _n2 = random.sample(nd1_U.difference(n2e[_n1]),1)[0]
                    except ValueError: continue
                    _e = (_n2,_n1)

            if _e not in _U:
                #--Updating edges
                neg_edges.add(_e)
                _U.add(_e)

                #--Updating node probs
                iis = np.searchsorted(nds,[_n1,_n2])
                probs[iis] =  [max([0,probs[i]-1]) for i in iis]
                #--Anotatting the new edge (to avoid repeating it)
                n2e[_n1].add(_n2)
                n2e[_n2].add(_n1)
                #--Updating tqdm bar
                pbar.update(1)

            else: #Just in case although I prevent existing edges to appear...
                continue
        pbar.close()

    #4) Final checking
    if len(neg_edges) > 0:
        #--Sorting
        neg_edges_sorted = np.unique(np.sort(list(neg_edges), axis=1),axis=0)

        #--Asserting that false edges are unique
        all_edges = set(map(tuple,np.sort(egs, axis=1)))
        assert len(set(map(tuple,neg_edges_sorted)) & all_edges) == 0
        #--Asserting that there are not any self-loop
        if allow_self_loops == False:
            assert np.all(neg_edges_sorted[:,0]!=neg_edges_sorted[:,1])

    return neg_edges

def network_stats(gx,weighted=False,ofile=None,dpi=150,warning_cutoff=0.5):
    """Given a networkx graph it calculates some stats realated to the connectivity of the metanodes"""

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import auc
    from matplotlib.ticker import ScalarFormatter
    class ScalarFormatterForceFormat(ScalarFormatter):
        def _set_format(self):  # Override function that finds format to use.
            self.format = "%1.3f"  # Give format here

    colpal = ['#fa6450', '#5078dc', '#78b43c', '#c864e1', '#fa9632', '#dcdadb']
    STATS = {}

    #Get metanode universes and degree distributions
    mnds, degree_d, weights = {}, {}, {}

    for n in gx.nodes:
        lb = gx.nodes[n]['label']
        if lb not in mnds:
            mnds[lb] = set([])
            degree_d[lb] = []
            if weighted:
                weights[lb] = []
        #--updating metanode universe
        mnds[lb].add(n)
        degree_d[lb].append(len(gx[n]))
        if weighted:
            w = [float(x['weight']) for x in gx[n].values()]
            weights[lb].append([np.percentile(w,PC) for PC in [0,25,50,75,100]])

    STATS['edges'] = gx.number_of_edges()
    STATS['mnds'] = {n:len(mnds[n]) for n in mnds}
    STATS['components'] = len(list(nx.connected_components(gx)))

    #Gini
    gini = gini_coef([len(i) for i in nx.connected_components(gx)])
    STATS['gini'] = gini

    #Degree
    STATS['degree'] = {}
    for lb,deg in degree_d.items():

        #--Degree distribution
        col_ix = 0
        plt.figure(figsize=(4, 4), dpi=dpi)
        sns.histplot(deg,label='Median: %i)'%(np.median(deg)), color=colpal[col_ix], alpha=0.7)
        col_ix+=1
        STATS['degree']['%s'%lb] = [np.mean(deg), np.min(deg),np.percentile(deg,25),np.percentile(deg,50),np.percentile(deg,75), np.max(deg)]
        title = '%s degree dist'%lb
        plt.xlabel('Degree')
        plt.ylabel('# of %s'%lb)
        plt.title(title)
        plt.legend()
        plt.yscale('log')

        if ofile:
            plt.savefig(ofile+'/%s.png'%'_'.join(title.split(' ')), bbox_inches='tight')
        else:
            plt.show()
        plt.close()

        #--degree count
        plt.figure(figsize=(6,4), dpi=dpi)
        title = '%s degree count'%lb
        deg = np.asarray(deg)
        bins = [1,2,3,4,5,10,20,50,100,500,1000]
        xtick_labels = ['1','2','3','4','5','10', '20', '50', '100','500']
        xtick_labels += ['{}\u00b3'.format(10), '>'+'{}\u00b3'.format(10)] #As the last bin is 1000 we can reduce its label

        r = [sum(deg<=b) for b in bins]
        for ix in range(len(r)):
            r[ix] = r[ix]- sum(r[:ix])
        r.append(sum(deg>bins[-1]))
        k = pd.DataFrame(zip(xtick_labels,r), columns=['Degree','Count'])
        sns.barplot(y='Count',x='Degree', data=k)

        plt.xticks(np.arange(len(bins)+1), xtick_labels,fontsize=fsize-1)
        plt.title(title)

        if ofile:
            plt.savefig(ofile+'/%s.png'%'_'.join(title.split(' ')), bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    #Weights
    if weighted:
        STATS['weight'] = {}

        for lb,mt in weights.items():
            mt = np.asarray(mt)
            STATS['weight']['%s'%lb] = []
            fig = plt.subplots(figsize=(20,4),dpi=dpi)
            axs = [plt.subplot(151), plt.subplot(152), plt.subplot(153), plt.subplot(154), plt.subplot(155)]

            names = ['Min','Q25','Q50','Q75','Max']
            for i in range(mt.shape[1]):
                z = mt[:,i]
                sns.histplot(z,color=colpal[i], alpha=0.7, ax=axs[i])
                STATS['weight']['%s'%lb].append([np.mean(mt[:,i]), np.min(mt[:,i]),np.percentile(mt[:,i],25),np.percentile(mt[:,i],50),np.percentile(mt[:,i],75), np.max(mt[:,i])])

                axs[i].set_ylabel('# of %s'%lb)
                axs[i].set_title(names[i])
                axs[i].set_yscale('log')
                if len(set(z)) < 6:
                    ticks = np.unique(z)
                else:
                    ticks = [x for x in np.arange(min(z),max(z), (max(z)-min(z))/5)]
                    ticks[0] = min(z)
                    ticks.append(max(z))
                axs[i].set_xticks(ticks)
                fmt = ScalarFormatterForceFormat()
                fmt.set_powerlimits((0,0))
                axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation = 35)
                axs[i].xaxis.set_major_formatter(fmt)

            title = '%s weight dist'%lb
            plt.suptitle(title, y=1.05, fontsize=fsize)

            plt.tight_layout()
            if ofile:
                plt.savefig(ofile+'/%s.png'%'_'.join(title.split(' ')), bbox_inches='tight')
            else:
                plt.show()
            plt.close()

    if STATS['components'] == 1:
        sys.stderr.write("--> The network only has 1 component\n")
        STATS['C1'] = {n:1 for n in mnds}
        STATS['aucs'] = {n:1 for n in mnds}
        STATS['gini'] =1
    else:
        #Plotting
        plt.figure(figsize=(4, 4), dpi=dpi)
        col_ix = 0
        STATS['aucs']  = {}
        STATS['C1'] = {}
        for n,u in mnds.items():
            v = sorted([len(u & x)/len(u) for x in nx.connected_components(gx)],reverse=True)
            cS = np.cumsum(v)
            cS = cS / np.max(cS)
            y = [0]+list(cS)
            x = np.arange(len(y))/len(y)
            AUC = auc(x,y)
            C1 = v[0]

            plt.plot(x,y, color=colpal[col_ix], linestyle="-", lw=1, label='%s (C1: %.2f)'%(n, C1))
            col_ix+=1

            STATS['aucs'][n] = auc(x,y)
            STATS['C1'][n] = C1

            if C1 < warning_cutoff:
                sys.stderr.write("""
                ***********************************************************************************
                **** WARNING: Only %.2f %s  are in the largest component (recommended: > %.2f) ****
                ***********************************************************************************
                """%(C1,n, warning_cutoff))

        #Total
        all_nds = list(mnds.values())[0].union(*list(mnds.values()))

        v = sorted([len(all_nds & x)/len(all_nds) for x in nx.connected_components(gx)],reverse=True)
        cS = np.cumsum(v)
        cS = cS / np.max(cS)
        y = [0]+list(cS)
        x = np.arange(len(y))/len(y)

        plt.plot(x,y, color='black', linestyle="--", lw=1, label='total (C1: %.2f)'%(v[0]))

        plt.xlabel('# of components (N: %i)'%STATS['components'])
        plt.ylabel('Proportion')

        x_ix = np.arange(1,min(21,len(x)),1)
        x = x[x_ix]
        plt.xlim(x[1],max(x))
        plt.xticks(x,x_ix,rotation=90)
        plt.legend(handlelength=0.8)
        plt.title('Nodes across components')
        if ofile:
            plt.savefig(ofile+'/stats_plot.png', bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    #Writing stats
    if ofile:
        with open(ofile+'/network_stats.txt','w') as o:

            #Edges
            o.write('>Edges:\n%i\n'%STATS['edges'])
            o.write('\n')

            #Metanodes
            o.write('>Metanodes:\n')
            for n in sorted(STATS['mnds']):
                o.write('%s: %i\n'%(n,STATS['mnds'][n]))
            o.write('\n')

            #Components
            o.write('>Components:\n%i\n'%STATS['components'])
            o.write('\n')

            #Degree
            o.write('>Degree:\n')
            for n in sorted(STATS['degree']):
                o.write('%s\n'%n)
                for r in list(zip(['mean','min','Q25','Q50','Q75','max'],STATS['degree'][n])):
                    if r[0] == 'mean':
                        o.write('\t%s: %.3f\n'%(r[0],r[1]))
                    else:
                        o.write('\t%s: %i\n'%(r[0],r[1]))
            o.write('\n')

            #Weights
            if weighted:
                o.write('>Weights:\n')
                for n in sorted(STATS['weight']):
                    o.write('%s\n'%n)
                    names = ['Min','Q25','Q50','Q75','Max']
                    for i in range(len(STATS['weight'][n])):
                        o.write('\t%s\n'%names[i])

                        for r in list(zip(['mean','min','Q25','Q50','Q75','max'],STATS['weight'][n][i])):
                            o.write('\t\t%s: %.2E\n'%(r[0],r[1]))

                o.write('\n')


            #C1
            o.write('>C1:\n')
            for n in sorted(STATS['C1']):
                o.write('%s: %.3f\n'%(n,STATS['C1'][n]))
            o.write('\n')

            #AUCS
            o.write('>Aucs:\n')
            for n in sorted(STATS['aucs']):
                o.write('%s: %.3f\n'%(n,STATS['aucs'][n]))
            o.write('\n')

            #GINI
            o.write('>Gini:\n%.3f\n'%STATS['gini'])
            o.write('\n')

    return STATS

def assess_network_quality(metaedge2dts, metapath, approach='merged',ofile=None,dpi=150, verbose=True):
    """
    Given a dictionary of metaedges with list of datasets and a metapath it builds a network, prune it and calculate some stats to validate the quality of the network.

    Returns
    -------
    networkx.Graph, dictionary with Stats.

    Keywords
    --------
    metaedge2dts -- dictionary of type 'metaedge':[dataset1,dataset2].
                    Ex: {'CPD-int-GEN':['drugbank_HMZ','ctdchemical_HMZ'],'GEN-ass-PWY':['reactome_cpdb']}
    metapath -- metapath to be calculated. The final metanode must be the same than the first one
                Ex: 'CPD-int-GEN-ass-PWY-ass-GEN-int-CPD'.
    approach -- How to process the repeated edges across datasets. Currently supports 2 flags:
                *merged* -> Repeated edges are removed
                *weighted* -> It returns a third column with the number of datasets reporting the edge. Useful if you want to prioritize the rwalk through edges reported by more than one dataset.
    ofile -- Path where the output files and figures will be saved.
    dpi -- dpi of the figures.
    """
    # Checking input variables
    metapath = copy.copy(metapath)
    if type(metapath) == str:
        metapath = metapath.split('-')

    # 1) Read data
    sys.stderr.write('1) Reading the data...\n************************\n')
    metaedge_dict = read_metaedges(metaedge2dts,approach= approach)

    # 2) Graph Pruning
    sys.stderr.write('\n2) Graph pruning...\n************************\n')
    metaedge_dict = network_pruning(metaedge_dict,metapath,write_to=ofile+'/pruning.txt',verbose=verbose)

    # 3) Building graph
    sys.stderr.write('\n3) Building the graph...\n************************\n')
    gx = build_graph(metaedge_dict)
    gx.name = '-'.join(metapath)

    # 4) Calculating stats
    stats = network_stats(gx,ofile=ofile,dpi=dpi)

    return gx, stats
