import sys
import os
import numpy as np
import pandas as pd
from collections import Counter
from itertools import chain
from scipy.spatial.distance import cdist
from tqdm import tqdm

distance = 'cosine'
cosine_distance_cutoff = 0.8 # Lower bound
bonafide_cutoff = 0.5 # Significance cutoff (pvale: 0.0005)


xref_path =  '../disease/'
families = ['DOID', 'EFO', 'HP', 'MEDDRA', 'MESH', 'OMIM', 'ORPHA']

def get_one_hot_vectors(document2terms,universe):
    d = {}
    for document, terms in tqdm(document2terms.items(),leave=False):
        r = np.zeros(len(universe))
        r[np.searchsorted(universe, list(terms))] = 1
        d[document] = r
    return d

def get_idf_hot_vector(document2terms,universe):
    #Getting number of documents
    N = len(document2terms)
    r = []
    for x in tqdm(document2terms.values(),leave=False):
        r+=list(set(x))
    d = {x:(1+np.log(N/y)) for x,y in Counter(r).items()}
    return np.asarray([d[x] for x in universe])

def tfidf_one_hot_array(document2terms,universe):
    r = get_one_hot_vectors(document2terms,universe)
    idf_vector = get_idf_hot_vector(document2terms,universe)
    return {x:r[x]*idf_vector for x in r}

def get_umls_xrefs_from_family(family):
    
    d = {}
    umls_universe = set([])
    with open(xref_path+'/disgenet.tsv','r') as f:
        f.readline()
        for l in f:

            h = l.rstrip().split('\t')
            n1,n2 = h[0],h[1]
            
            if n1.startswith('UMLS:'):
                umls_universe.add(n1)
                
                if n2.startswith(family):
                    if n2 not in d:
                        d[n2] = set([])
                    d[n2].add(n1)
                

            if n2.startswith('UMLS:'):
                umls_universe.add(n2)
                
                if n1.startswith(family):
                    if n1 not in d:
                        d[n1] = set([])
                    d[n1].add(n2)
           
    #Only adding those MEDDRA with UMLS in common with the rest of vocabularies
    if family == 'MEDDRA':
        with open(xref_path+'/meddra.tsv','r') as f:
            f.readline()
            for l in f:
                h = l.rstrip().split('\t')
                n1,n2 = h[0],h[1]

                if n1 in umls_universe:
                    if n2 not in d:
                        d[n2] = set([])
                    d[n2].add(n1)

                if n2 in umls_universe:
                    if n1 not in d:
                        d[n1] = set([])
                    d[n1].add(n2)

    return d, umls_universe


sys.stderr.write('Getting tfidf vectors...\n')
with open(xref_path+'/full_tfidf_umls_distance.tsv','w') as o:
    o.write('n1\tn2\tumls_tdidf_dist\n')
    
    for i1 in tqdm(range(len(families))):
        f1 = families[i1]
        d1, umls_universe1 = get_umls_xrefs_from_family(f1)

        for i2 in tqdm(range(i1+1,len(families)), leave=False):
            f2 = families[i2]
            d2, umls_universe2 = get_umls_xrefs_from_family(f2)

            #--Final D and umls universe
            d =  {**d1, **d2}
            umls_universe = umls_universe1 & umls_universe2
            d = {x:d[x]&umls_universe for x in d.keys() if len(d[x]&umls_universe)>0}
            umls_universe = sorted(set(chain.from_iterable(d.values())))

            #--Getting vectors
            r = tfidf_one_hot_array(d,umls_universe)

            #--Transfoming into matrices
            dis1 = np.unique([x for x in r if x in d1])
            dis2 = np.unique([x for x in r if x in d2])
            v1 = np.array([r[x] for x in dis1])
            v2 = np.array([r[x] for x in dis2])
            
            #--Cleaning up
            del r, d, umls_universe, d2, umls_universe2

            #--Getting distance
            dist = cdist(v1,v2,distance)

            #Getting those pairs below the cutoff
            ixs1,ixs2 = np.where(dist<cosine_distance_cutoff)

            m = sorted((zip(dis1[ixs1],dis2[ixs2], dist[ixs1,ixs2])))

            for n1,n2,sc in m:
                o.write('%s\t%s\t%.4f\n'%(n1,n2,sc))
                
            del dist, ixs1, ixs2, m

#--Creating a file with bonafide predictions (p-value 0.0005)
with open(xref_path+'/full_tfidf_umls_distance.tsv','r') as f,open(xref_path+'/tfidf_umls_distance.tsv', 'w') as o:
    o.write(f.readline())
    for l in f:
        h = l.rstrip().split('\t')
        if float(h[-1]) < bonafide_cutoff:
            o.write(l)
       
