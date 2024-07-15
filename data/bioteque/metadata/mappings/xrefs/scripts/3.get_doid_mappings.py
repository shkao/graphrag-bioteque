# Getting cross-references
# Here I get the cross references ("gold standard mappings") to map from any available vocabulary in the graph to a refence vocabulary. For that, I generate a file for each vocabulary where all the mappings from different vocabularies (n1) are mapped to the reference (n2).

import sys
import os
import numpy as np
root = '../../../../'

N_path = root+'/graph/nodes/Disease.tsv'
xrf_path = '../disease/'
OPATH = root+'/metadata/mappings/DIS/'
if not os.path.isdir(OPATH):
    os.mkdir(OPATH)

# ## Disease
vocabularies = ['doid','hp','mesh','omim','efo','umls','orpha','meddra']


vcb2nd = {x:set([]) for x in vocabularies}
with open(N_path,'r') as f:
    ix = np.where(np.asarray(f.readline().strip('\n').split('\t')) == 'vocabulary')[0][0]
    for l in f:
        h = l.rstrip('\n').split('\t')
        vcb2nd[h[ix]].add(h[0])
        

#Reference xref files for each vocabulary in Disease
reference_dts = {
    'doid':set(xrf_path+x for x in ['doid.tsv','disgenet.tsv']),
   # 'hp':set(xrf_path+x for x in ['hpo.tsv','disgenet.tsv']),
   # 'mesh':set(xrf_path+x for x in ['ctd_medic.tsv','disgenet.tsv']),
   # 'omim':set(xrf_path+x for x in ['ctd_medic.tsv','disgenet.tsv']),
   # 'efo':set(xrf_path+x for x in ['efo.tsv','disgenet.tsv']),
   # 'umls':set(xrf_path+x for x in ['disgenet.tsv']),
   # 'orpha':set(xrf_path+x for x in ['orphanet.tsv','disgenet.tsv']),
   # 'meddra':set(xrf_path+x for x in ['meddra.tsv'])
}
#Dataset that are predictions. Only added if they were not covered by the reference dts
predicted_dts = {
    'doid':set(xrf_path+x for x in ['tfidf_umls_distance.tsv']),
   # 'hp':set(xrf_path+x for x in ['tfidf_umls_distance.tsv']),
   # 'mesh':set(xrf_path+x for x in ['tfidf_umls_distance.tsv']),
   # 'omim':set(xrf_path+x for x in ['tfidf_umls_distance.tsv']),
   # 'efo':set(xrf_path+x for x in ['tfidf_umls_distance.tsv']),
   # 'umls':set(xrf_path+x for x in ['tfidf_umls_distance.tsv']),
   # 'orpha':set(xrf_path+x for x in ['tfidf_umls_distance.tsv']),
   # 'meddra':set(xrf_path+x for x in ['tfidf_umls_distance.tsv'])
}


for vb in vocabularies:
    xrfs = {}
    xrfs_prd = {}
    
    #Reading reference files
    evidence = 'curated'
    sc = 1
    for file in reference_dts[vb]:
        with open(file,'r') as f:
            f.readline()
            for l in f:
                h = l.rstrip('\n').split('\t')
                n1,n2 = h[0],h[1]
                                
                if n1 in vcb2nd[vb] and n2 in vcb2nd[vb]: 
                    continue
                elif n1 in vcb2nd[vb]:
                    xrfs[(n2,n1)] ={'evidence':evidence,'score':sc}
                elif n2 in vcb2nd[vb]:
                    xrfs[(n1,n2)] ={'evidence':evidence,'score':sc}
                else:
                    continue

    #Reading prediction files
    evidence = 'predicted'
    for file in predicted_dts[vb]:
         with open(file,'r') as f:
            f.readline()
            for l in f:
                h = l.rstrip('\n').split('\t')
                n1,n2,sc = h[0],h[1],1-float(h[2]) #Transforming cosine distance into similarity
                if sc == 1:
                    sc = int(sc)
                
                if n1 in vcb2nd[vb] and n2 in vcb2nd[vb]: 
                    continue
                elif n1 in vcb2nd[vb]:
                    p = (n2,n1)
                elif n2 in vcb2nd[vb]:
                    p = (n1,n2)
                else:
                    continue
                    
                if p in xrfs:
                    continue
                else:
                    xrfs_prd[p] = {'evidence':evidence,'score':sc}
    
    #Updating xrefs with the predicted xrefs
    assert len(set(xrfs)&set(xrfs_prd)) == 0
    xrfs.update(xrfs_prd)
    
    #Writing Disease gold standard xrefs
    with open(OPATH+'/%s.tsv'%vb,'w') as o:
        o.write('n1\tn2\tevidence\tscore\n')
        for pair,data in xrfs.items():
            o.write('%s\t%s\t%s\t%s\n'%(pair[0],pair[1],xrfs[pair]['evidence'],str(np.round(xrfs[pair]['score'],4))))
