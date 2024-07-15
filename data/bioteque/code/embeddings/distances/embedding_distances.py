root_path = '../../../'
code_path = root_path+'code/embeddings/'
emb_path = root_path+'embeddings/'

import sys
import pandas as pd
import numpy as np
import h5py
from bisect import bisect_left
from scipy.stats import gmean
import faiss
from tqdm import tqdm
sys.path.insert(1, code_path)
from utils.embedding_utils import read_embedding, get_embedding_universe, get_faiss_index

def create_faiss_index(m, distance='cosine', write_to=None, dim=None):

    _m = np.asarray(m)
    if dim == None:
        dim = _m.shape[1]
    if distance == 'euclidean':
        index = faiss.IndexFlatL2(dim)
        index.add(_m)

    elif distance == 'cosine':
        index = faiss.IndexFlatIP(dim)
        m_norm = np.linalg.norm(_m, axis=1)
        index.add(_m / m_norm[:, None])

    if write_to:
        faiss.write_index(index,write_to)

    return index

def get_nn_from_faiss(faiss_index,v,k=5):

    _v = np.asarray(v)
    if len(v.shape) == 1:
        _v = _v[:,None]

    if faiss_index.metric_type == 0: #1 == euclidean | 0 == cosine
        _v = _v/np.linalg.norm(_v,axis=1)[:,None]

    return faiss_index.search(_v,k)

def measure_pvalues_from_edges(edges, metapath, dt, distance ='cosine', max_nneigh=0.1, average=True, normalize_by='universe',
                               strict_when_average=False, return_distance = False,  mnd1=None, mnd2=None, w=5, pp=False,
                               emb_path = emb_path):

    """
    Given a set of edges and a metapath embedding it measures the pvalue distances between the edges according to the metapath embedding

    Arguments

    edges (list) -- List of edges. Must have same ids than in the metapath
    metapath (str) -- Metapath name of the embedding
    dt (str) -- Dataset name of the embedding
    distances (str) -- Distance to use. Can be 'cosine' or 'euclidean' (default: 'cosine')
    max_nneigh (float, int or 'all') -- Proportion of raw number of neighbors to fetch. If 'all' it takes all the neighbors (default: 0.1 #10%)
    average (float) -- If true it returns the geometric mean of both p-values
    normalize_by (str) -- How to normalize the rankings. (default: 'universe')
                          If 'universe' it divides by the total number of neibhours.
                          If 'max_nneigh' it normalize by the max_nneigh taken.
                          If None it does not normalize.
    strict_when_average (boolean) -- If True it will set the maximum ranking value to an edge when one of the pvalues is maximum (default:False)
    return_distance (boolean) -- If True it also return the distance between the edges (default: False)
    mnd1 (str, None) -- Metanode corresponding to the first edge column (e.g. 'GEN'). If None it assumes is the same than the 1st mnd of the metapath (default:None)
    mnd2 (str, None) -- Metanode corresponding to the second edge column (e.g. 'GEN'). If None it assumes is the same than the 2nd mnd of the metapath (defualt:None)
    """

    #--Parsing inputs
    if type(edges[0]) not in [list, set, np.ndarray]:
        edges = np.array([edges])
    edges = np.asarray(list(edges))

    if type(metapath) != str and type(metapath) != np.str_:
        mpath = '-'.join(metapath)
    else:
        mpath = metapath

    if not mnd1:
        mnd1 = mpath.split('-')[0]

    if not mnd2:
        mnd2 = mpath.split('-')[-1]

        get_all = True
    else:
        get_all = False

    if pp is None:
        _mp = mpath.split('-')
        if _mp[0] == _mp[-1]:
            pp = False
        else:
            if mnd1 != mnd2:
                pp = False
            else:
                pp = True

    if distance not in ['euclidean', 'cosine']:
        sys.exit('Distance must be either "cosine" or "euclidean"\n')

    #--Get node label lists
    lbs1 = np.array(get_embedding_universe(mpath= mpath, dt= dt, mnd= mnd1, emb_path= emb_path))
    lbs2 = np.array(get_embedding_universe(mpath= mpath, dt= dt, mnd= mnd2, emb_path= emb_path)) if mnd1 != mnd2 else lbs1

    #--Mapping edges to positions in universe
    edges = np.array([[bisect_left(lbs1, e[0]), bisect_left(lbs2, e[1])] for e in edges])

    #--Set number of neighbors to retrieve
    if max_nneigh == 'all':
        K = len(lbs1)
    elif isinstance(max_nneigh, float) or isinstance(max_nneigh, np.floating):
        K = int(len(lbs1)*max_nneigh)

    #--Get rank mnd1
    m1 = read_embedding(mpath=mpath, dt=dt, w=w, pp=pp, mnd=mnd1,
                        just_the_matrix=True, emb_path= emb_path)

    faiss_ix1 =  get_faiss_index(mpath=mpath, dt=dt, w=w, pp=pp, mnd=mnd2,
                                 distance=distance, emb_path= emb_path)

    ds1, rk1 = get_nn_from_faiss(faiss_ix1, m1, k=K)
    del ds1, faiss_ix1

    #--Get rank mnd2
    if len(lbs1) == len(lbs2) and np.all(lbs1 == lbs2):
        rk2 = rk1
        m2 = m1
    else:
        m2 = read_embedding(mpath=mpath, dt=dt,w=w,pp=pp, mnd=mnd2,
                            just_the_matrix=True, emb_path= emb_path)
        faiss_ix2 =  get_faiss_index(mpath=mpath, dt=dt, w=w,  pp=pp, mnd=mnd1,
                                     distance=distance, emb_path= emb_path)
        ds2, rk2 = get_nn_from_faiss(faiss_ix2, m2, k=K)
        del ds2, faiss_ix2

    #--Setting normalize factor (if specified)
    if normalize_by == 'universe':
        max_values = [len(lbs2), len(lbs1)]
    elif normalize_by == 'max_nneigh':
        max_values = [rk1.shape[1]+1, rk2.shape[1]+1]
    else: #<-- Same as max nneigh
        max_values = [rk1.shape[1]+1, rk2.shape[1]+1]

    #Getting distances
    if return_distance:

        #--Making distance function
        import scipy.spatial.distance as scipy_dist
        dist_func = scipy_dist.cosine if distance == 'cosine' else scipy_dist.euclidean
        def compute_dist(e, m1, m2, dist_func):
            return dist_func(m1[e[0]], m2[e[1]])

        #--Getting distances
        distances = np.apply_along_axis(compute_dist, 1, edges,
                                    m1=m1,m2=m2, dist_func=dist_func)
    del m1, m2

    #Getting pvalues

    #--Making pvalue function
    def compute_pvalue(e, rks, max_values, pseudocount=True):
        v = [np.where(rks[0][e[0]] == e[1])[0], np.where(rks[1][e[1]] == e[0])[0]]
        v[0] = v[0][0]+pseudocount if len(v[0])> 0 else max_values[0]
        v[1] = v[1][0]+pseudocount if len(v[1])> 0 else max_values[1]
        return v

    #--Getting pvalues
    pvalues = np.apply_along_axis(compute_pvalue, 1, edges,
                                rks=[rk1,rk2],
                                max_values=max_values).astype(np.float)
    #--Applying normalizations
    if normalize_by not in ['', None, np.nan]:
        pvalues[:,0] = pvalues[:,0]/max_values[0]
        pvalues[:,1] = pvalues[:,1]/max_values[1]
    if average:
        if strict_when_average:
            if normalize_by not in ['', None, np.nan]:
                pvalues[pvalues==1] = np.nan
                pvalues = np.apply_along_axis(gmean, 1, pvalues)
                pvalues[pd.isnull(pvalues)] = 1
            else:
                pvalues[pvalues==max_values] = np.nan
                pvalues = np.apply_along_axis(gmean, 1, pvalues)
                pvalues[pd.isnull(pvalues)] = int(np.round(gmean(max_values)))
        else:
            pvalues = np.apply_along_axis(gmean, 1, pvalues)

    if return_distance:
        return np.asarray(pvalues), np.asarray(distances)
    else:
        return np.asarray(pvalues)
