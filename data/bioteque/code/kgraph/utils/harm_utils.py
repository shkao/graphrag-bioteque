import requests
import os
import zlib
import sys
import numpy as np

def _download_file(response, filename):
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)

def _download_and_decompress_file(response, filename):
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    filename = filename[:-3]
    with open(filename, 'w+') as f:
        while True:
            chunk = response.raw.read(1024)
            if not chunk:
                break

            string = decompressor.decompress(chunk).decode("utf-8",errors='ignore')
            f.write(string)


def download_datasets(selected_datasets, selected_downloads, decompress=False):
    for dataset, path in selected_datasets:
        if not os.path.exists(dataset):
            os.mkdir(dataset)

        for downloadable in selected_downloads:
            url = 'http://amp.pharm.mssm.edu/static/hdfs/harmonizome/data/%s/%s' %\
                  (path, downloadable)
            response = requests.get(url, stream=True)
            filename = '%s/%s' % (dataset, downloadable)

            # Not every dataset has all downloadables.
            if response.status_code != 200:
                continue

            if decompress and 'txt.gz' in filename:
                _download_and_decompress_file(response, filename)

            else:
                _download_file(response, filename)

        print('%s downloaded.' % dataset)


def file2matrix(p,attribute_index=3,gene='id',meta_data=False):
    """
    Reads a file and return a edge-list matrix.

    If meta data, it also returns the column information and a boolean
    (True if the matrix is binary (weights = 1 / -1), False if it's not)

    If attribute_index is an interable, it returns multiple ids separeted by: ||
    """

    if gene == 'id':
        gn_idx = 2
    elif gene=='name':
        gn_idx = 0
    else:
        sys.exit('Incorrect gene')

    with open(p,'r') as f:

        f.readline()# header_model

        #Reading header
        hd = f.readline().rstrip().split('\t')
        d = {'source': hd[0],'source_desc':hd[1], 'source_id':hd[2], 'target':hd[3],
             'target_desc': hd[4], 'target_id': hd[5], 'weight': hd[6], 'binary':False }

        #Making the matrix
        m = []
        for l in f:
            h = l.rstrip().split('\t')

            s = h[gn_idx].rstrip() #source (gene symbol)
            if type(attribute_index) == int:
                t = h[attribute_index].rstrip() #target (the atributte)
           #If attribute_index is an interable --> concatenating multiple ids using '||'
            else:
                t = ''
                for i in attribute_index:
                    t+=h[i].strip()+'||'
                t = t.rstrip('||')
            w = float(h[6]) #weight
            if int(w) != 1:
                d['binary'] = True

            m.append([s,t,w])

    if meta_data:
        return np.array(m,dtype=object), d
    else:
        return np.array(m,dtype=object)

def split_binary(m,weights=False):
    """
    Returns 2 matrix separating the binary values (1,-1). If Weights it also return the weights.
    """
    if type(m) != np.ndarray:
        m = np.array(m,dtype=object)

    pos_id = np.where(m[:,-1]>0)[0]
    neg_id = np.where(m[:,-1]<0)[0]

    if weights:
        return m[pos_id], m[neg_id]
    else:
        return m[:,:-1][pos_id],m[:,:-1][neg_id]
