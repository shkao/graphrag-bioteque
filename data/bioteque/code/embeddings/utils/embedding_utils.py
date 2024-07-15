root_path = '../../../'
emb_path = root_path+'embeddings/'

import sys
import os
import numpy as np
import h5py
import pickle
import random
from collections import defaultdict
from bisect import bisect_left
from scipy.spatial.distance import pdist,squareform

def read_embedding(final_path=None,mpath=None,dt=None, mnd=None, pp=False, w=5, universe=None, just_the_matrix = False,
                   name = 'm', emb_path=emb_path):
    """
    Given a embedding path it returns a dictionary with all the embeddings.
    The file must be an h5 file with 'm' as the embedding matrix and 'rows' as np.array with the labels
    If final_folder is True it interpretates that the path is already the final folder and the ignores the rest of metapath2vev parameters
    If mnd is None it interpretates that the path is already the h5 file to be read.
    """

    if final_path:
        h5_file = final_path
        row_file = '/'+'/'.join([x for x in final_path.split('/') if x != ''][:-3]) + '/nodes/%s.txt'%(final_path.split('/')[-1])[:-3]
    else:
        if mpath is None:
            sys.exit('You must specify a metapath')
        if dt is None:
            sys.exit('You must specify a dataset')

        if type(mpath) not in [str, np.str_]:
            _mpath = '-'.join(list(mpath))
        else:
            _mpath = mpath

        if type(w) == int:
            w = str(w)

        if mnd is None:
            mnd = _mpath.split('-')[0][:3]

        #Checking whether the emb path has mnd has a prefix or not
        prefix = _mpath.split('-')[0][:3]
        if not os.path.exists(emb_path+prefix) and os.path.exists(emb_path+_mpath):
            prefix = ''

        h5_file = emb_path+'/%s/%s/%s/embs/w%s/%s.h5'%(prefix,_mpath,dt,w,mnd)
        row_file =  emb_path+'/%s/%s/%s/nodes/%s.txt'%(prefix,_mpath,dt,mnd)

    if pp == True and name=='m':
        name = 'm++'

    with open(row_file,'r') as f:
        rows = np.asarray(f.read().splitlines())

    with h5py.File(h5_file,'r') as f:
        if universe is not None:
            from bisect import bisect_left
            universe = sorted(set(rows) & set(universe))
            ixs = [bisect_left(rows, x) for x in universe]
            m = f[name][ixs]
            rows = rows[ixs]
        else:
            m = f[name][:]

    if just_the_matrix:
        return m
    else:
        return  dict(zip(rows,m))

def get_embedding_universe(final_path=None,mpath=None,dt=None, mnd=None, split_by_mnd=False,
                   emb_path=emb_path):
    """
    Given a embedding path it returns a all the rows labels
    """

    #--Checking which metanode needs to be returned
    if mnd is None:
        if mpath is not None:
            mnds = [mpath.split('-')[0],mpath.split('-')[-1]]
        else:
            _ = final_path.rstrip('/').split('/')[-2]
            mnds = [_[:3],_[-3:]]
    elif type(mnd) in [str, np.str_]:
        mnds = [mnd]
    else:
        mnds = mnd

    if mnds[0] == mnds[-1]:
        mnds = mnds[:1]

    #--Checking with input is given
    if final_path:
        row_file = '/'+'/'.join([x for x in final_path.split('/') if x != '']) + '/nodes/'
    else:
        if mpath is None:
            sys.exit('You must specify a metapath')
        if dt is None:
            sys.exit('You must specify a dataset')

        if type(mpath) not in [str, np.str_]:
            _mpath = '-'.join(list(mpath))

        else:
            _mpath = mpath

        #Checking whether the emb path has mnd has a prefix or not
        prefix = _mpath.split('-')[0][:3]
        if not os.path.exists(emb_path+prefix) and os.path.exists(emb_path+_mpath):
            prefix = ''

        row_file =  emb_path+'/%s/%s/%s/nodes/'%(prefix,_mpath,dt)

    #--Getting universe
    uv = []
    for mnd in mnds:
        with open(row_file+'/%s.txt'%mnd,'r') as f:
            if split_by_mnd:
                uv.append(f.read().splitlines())
            else:
                uv.extend(f.read().splitlines())
    return uv

def get_node2network_component(final_path=None, mpath=None, dt=None,
                   emb_path=emb_path):

    if final_path:
        path = '/'+'/'.join([x for x in final_path.split('/') if x != ''][:-3]) + '/nd2st.h5'
    else:
        if mpath is None:
            sys.exit('You must specify a metapath')
        if dt is None:
            sys.exit('You must specify a dataset')

        if type(mpath) != str and type(mpath) != np.str_:
            _mpath = '-'.join(list(mpath))
        else:
            _mpath = mpath


        #Checking whether the emb path has mnd has a prefix or not
        prefix = _mpath.split('-')[0][:3]
        if not os.path.exists(emb_path+prefix) and os.path.exists(emb_path+_mpath):
            prefix = ''
        path =  emb_path+'/%s/%s/%s/nd2st.h5'%(prefix,_mpath,dt)

    with h5py.File(path, 'r') as f:
        return  dict(zip(f['nd'][:].astype(str), f['component'][:].astype(np.int)))

class RNDuplicates():
    """RNDuplicates class."""

    def __init__(self, nbits=16, only_duplicates=False, min_bucket_size=10,
                 check_distance=True, distance_to_check='cosine', max_median_dist=0.5,
                 cpu=1):
        """Initialize a RNDuplicates instance.

        Args:
            nbits (int): Number of bits to use to quantize.
            only_duplicates (boolean): Remove only exact duplicates.
            min_bucket_size (int): Minimum number of similar elements to consider the removing
            check_distance (boolean): If consider the within bucket distance to remove the elements
            distance_to_check (str): Distance to be used when check_distance is True. Can be any accepted by scipy.spatial.distance.pdist
            max_median_dist (float): Maximum median distance cutoff that an element must have in order to be considered "close". Requires check distance=True.
            cpu (int): Number of cores to use.
        """
        self.nbits = nbits
        self.only_duplicates = only_duplicates
        self.min_bucket_size = min_bucket_size
        self.check_distance = check_distance
        self.distance_to_check = distance_to_check
        self.max_median_dist = max_median_dist
        self.cpu = cpu
        self.threshold = 100000
        self.chunk = 1000
        self.data_file = ''

        sys.stderr.write('RNDuplicates to use ' + str(self.nbits) + " bits"+'\n')

    def remove(self, data, keys=None, save_dest=None, just_mappings=False, just_keys=False):
        """Remove redundancy from data.

        Args:
            data (array): The data to remove duplicates from. It can be a numpy
                array or a file path to a ``HDF5`` file with dataset ``m``.
            keys (array): Array of keys (rows) for the input data. If `None`, rows are
                taken from ``HDF5`` dataset ``rows``.
            save_dest (str): If the result needs to be saved in a file,
                the path to the file. (default: None)
            just_mappings (bool): Just return the mappings. Only applies if
                save_dest is None. (default=False)
            just_keys (bool): Just return the final keys (not removed). Only applies if
                save_dest and just_mappings are None. (default=False)
        Returns:
            keys (array):
            data (array):
            mappings (dictionary):

        """
        try:
            import faiss
        except ImportError:
            raise ImportError("requires faiss " +
                              "https://github.com/facebookresearch/faiss")

        faiss.omp_set_num_threads(self.cpu)

        if type(data) == str:
            sys.stderr.write("Data input is: " + data+'\n')
            if os.path.isfile(data):
                dh5 = h5py.File(data, 'r')
                if "m" not in dh5.keys():
                    raise Exception(
                        "H5 file does not contain dataset 'm'")
                data_size = dh5["m"].shape
                if (data_size[0] < self.threshold and data_size[1] < self.threshold) or self.only_duplicates:
                    self.data = np.array(dh5["m"][:], dtype=np.float32)
                else:
                    self.data = None
                    self.data_file = data

                    if self.check_distance is True:
                        sys.stderr.write('Warning --> Too many rows to check_distances (total:%i, max:%i). Skipping check_distance\n'%(data_size[0], self.threshold))
                        self.check_distance = False


                self.data_type = dh5["m"].dtype

                self.keys = dh5["rows"][:] if 'rows' in dh5.keys() else np.arange(data_size[0])
                dh5.close()

            else:
                raise Exception("This module only accepts .h5 files\n")

        else:
            self.data = data
            data_size = self.data.shape
            self.data_type = data.dtype
            if keys is None:
                self.keys = np.array(range(len(data)))
            else:
                self.keys = np.array(keys)

        sys.stderr.write("Size before removing: " + str(data_size[0])+'\n')

        self.final_ids = list()
        self.mappings = dict()

        if self.only_duplicates:
            indexl2 = faiss.IndexFlatL2(self.data.shape[1])

            indexl2.add(self.data)

            sys.stderr.write("Done adding in L2 space\n")

            D, I = indexl2.search(self.data, 1000)

            sys.stderr.write("Done searching in L2 space\n")

            done = set()

            for i in range(len(D)):
                if i in done:
                    continue
                indexes = []
                for j in range(1000):
                    if i == I[i][j]:
                        continue
                    if D[i][j] <= 0.0:
                        done.add(I[i][j])
                        indexes.append(I[i][j])
                    else:
                        if len(indexes) > 0:
                            chosen = random.choice(indexes)
                            self.final_ids.append(chosen)
                            for v in indexes:
                                self.mappings[v] = self.keys[chosen]
                        else:
                            self.final_ids.append(i)
                            self.mappings[self.keys[i]] = self.keys[i]

                        break

        else:

            indexlsh = faiss.IndexLSH(data_size[1], self.nbits)

            if self.data is None:

                starts = range(0, data_size[0], self.chunk)

                dh5 = h5py.File(self.data_file, 'r')

                for start in starts:

                    indexlsh.add(
                        np.array(dh5["m"][start:start + self.chunk], dtype=np.float32))
                dh5.close()

            else:

                indexlsh.add(self.data)

            indexes = faiss.vector_to_array(
                indexlsh.codes).reshape(-1, int(indexlsh.nbits / 8))

            buckets = defaultdict(list)

            for i in range(len(indexes)):
                buckets[indexes[i].tobytes()].append(i)

            for key, value in buckets.items():

                if(len(value) > self.min_bucket_size):

                    if self.check_distance: # --> Skips values that are not similar enough (i.e. median(dist) > dist_cutoff)
                        med = np.median(np.sort(squareform(pdist(self.data[value], self.distance_to_check)),axis=1)[:,1:], axis=1)
                        unsimilar_values = set(np.asarray(value)[med > self.max_median_dist])
                        for v in unsimilar_values:
                            self.final_ids.append(v)
                            self.mappings[self.keys[v]] = self.keys[v]
                        value = [x for x in value if x not in unsimilar_values]

                    if len(value) > 0: #--> Just in case after checking distances the bucket got empty
                        chosen = random.choice(value)
                        self.final_ids.append(chosen)
                        for v in value:
                            self.mappings[self.keys[v]] = self.keys[chosen]
                else:
                    for v in value:
                        self.final_ids.append(v)
                        self.mappings[self.keys[v]] = self.keys[v]

        self.final_ids.sort()

        sys.stderr.write("Size after removing: " + str(len(self.final_ids))+'\n')
        if save_dest is not None:
            self.save(save_dest)
        else:
            if just_mappings:
                return self.mappings
            elif just_keys:
                return self.keys[np.array(self.final_ids)]
            else:
                if self.data is None:
                    dh5 = h5py.File(self.data_file, "r")
                    self.data = dh5["V"][:]
                    dh5.close()
                return self.keys[np.array(self.final_ids)], np.array(self.data[np.array(self.final_ids)], dtype=self.data_type), self.mappings

    def save(self, destination):
        """Save non-redundant data.

        Save non-redundant data to a ``HDF5`` file.

        Returns:
            destination (str): The destination file path.
        """

        dirpath = os.path.dirname(destination)

        sys.stderr.write("Saving removed duplicates to : " + destination+'\n')
        list_maps = sorted(self.mappings.items())
        sys.stderr.write("Starting to write to : " + destination+'\n')
        with h5py.File(destination, 'w') as hf:
            keys = self.keys[np.array(self.final_ids)]
            hf.create_dataset("rows", data=np.array(keys, dtype='S'))
            if self.data is None:
                dh5 = h5py.File(self.data_file, 'r')
                V = np.array(
                    [dh5["m"][i] for i in self.final_ids], dtype=self.data_type)
            else:
                V = np.array(
                    self.data[np.array(self.final_ids)], dtype=self.data_type)
            hf.create_dataset("m", data=V)
            hf.create_dataset("shape", data=V.shape)
            hf.create_dataset("mappings",
                              data=np.array(list_maps,dtype='S'))
        sys.stderr.write("Writing mappings to " + dirpath+'\n')
        with open(os.path.join(dirpath, "mappings"), 'wb') as fh:
            pickle.dump(self.mappings, fh)

def get_faiss_index(final_path=None,mpath=None,dt=None, mnd=None, pp=False, w=5, distance='euclidean',
                   emb_path=emb_path):
    """
    Given a embedding path it returns a dictionary with all the embeddings.
    The file must be an h5 file with 'm' as the embedding matrix and 'rows' as np.array with the labels
    If final_folder is True it interpretates that the path is already the final folder and the ignores the rest of metapath2vev parameters
    If mnd is None it interpretates that the path is already the h5 file to be read.
    """
    import faiss

    if final_path:
        index_file = final_path
    else:
        if mpath is None:
            sys.exit('You must specify a metapath')
        if dt is None:
            sys.exit('You must specify a dataset')

        if pp == True:
            pp = '+'
        else:
            pp = ''

        if type(mpath) not in [str, np.str_]:
            _mpath = '-'.join(list(mpath))
        else:
            _mpath = mpath

        if type(w) == int:
            w = str(w)

        if mnd is None:
            mnd = _mpath.split('-')[0]

        if distance =='euclidean':
            ff = '%s.l2.faiss'%mnd
        elif distance == 'cosine':
            ff = '%s.cos.faiss'%mnd
        else:
            sys.exit('Distance %s is not available\n'%distance)

        #Checking whether the emb path has mnd has a prefix or not
        prefix = _mpath.split('-')[0][:3]
        if not os.path.exists(emb_path+prefix) and os.path.exists(emb_path+_mpath):
            prefix = ''

        index_file = emb_path+'/%s/%s/%s/nneigh/%sw%s/%s'%(prefix,_mpath,dt,pp,w,ff)

        try:
            faiss_index = faiss.read_index(index_file)
        except RuntimeError:
            sys.exit('No faiss index file was found in: %s\n'%index_file)

    return faiss_index


def map_matrix2emb(matrix,embedding_list, output_dtype=np.float32):
        """
        Maps m matrix labels to embeddings.
        ALL labels in m must have an embedding inside the specific embedding path.

        Keyargs:
        m -- Numpy matrix with the labels
        embedding_list -- list of embeddings. Can be path (to h5 file embeddings) or dictionaries {label:[emb]}
        """

        m = np.asarray(matrix, dtype=str)
        r = []
        if type(embedding_list[0]) in [str, np.str_]:

            def index_left(a, x):
                'Locate the leftmost value exactly equal to x'
                if type(x) in [str, np.str_] or type(x) == float or type(x) == int:
                    return bisect_left(a, x)
                else:
                    return [bisect_left(a, i) for i in x]

            for i in range(len(embedding_list)):
                E = embedding_list[i]
                rows = read_embedding_universe(final_path = E)
                v = np.unique(m[:,i]).astype(str) #Unique and sorted!

                #To avoid misleadings in index_left ALL X labels must be in the embedding file
                try:
                    assert set(v).issubset(set(rows))
                except:
                    sys.exit('There are labels without embedding representations: %s'%embedding_list[i])

                ixs = index_left(rows,v)

                with h5py.File(E,'r') as f:
                    emb = f['m'][ixs]

                r.append(emb[index_left(v,m[:,i].astype(str))])

        else:
            for i in range(len(embedding_list)):
                r.append([embedding_list[i][x] for x in m[:,i]])

        return np.asarray([np.concatenate([r[j][i] for j in range(len(r))]).astype(output_dtype) for i in range(len(r[0]))])
