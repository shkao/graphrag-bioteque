import sys
import os
from shutil import rmtree
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, './utils/')
import ontology_processing as DSP
import mappers as mps

onto_path = '../../metadata/ontologies/inheritable/'
node_path = '../../graph/nodes/'
mappings_path = '../../metadata/mappings/' 
out_path = '../../graph/processed/'
inp_path = '../../graph/raw/'

# *************
#   Functions
# *************
def get_node_universe(mnd, node_path = node_path):

    name = mps.node2abbr(mnd, reverse=True) if len(mnd) == 3 else mnd
    nd_uv = pd.read_csv(node_path+'/%s.tsv.gz'%name, sep='\t')
    return set(nd_uv['code'][nd_uv['is_reference']==True])

def remove_redundancy(df, undirected=False):

    #--Sort if undirected to remove duplicates
    if undirected:
        df.values.sort(axis=1)

    #--Remove duplicates
    df = df.drop_duplicates()

    #--Remove self loops
    df = df[np.array(df['n1'] != df['n2'])]

    #--Reset index
    df = df.reset_index(drop=True)

    return df

def get_dis2doid():
    with open(mappings_path+'/DIS/doid.tsv','r') as f:
        next(f)

        d2doid = {}
        for l in f:
            h = l.rstrip('\n').split('\t')
            if h[0] not in d2doid:
                d2doid[h[0]] = set([])
            d2doid[h[0]].add(h[1])

    return d2doid

def map_diseases(df, mnd1, mnd2, propagate=True):

    d2doid = get_dis2doid()

    #--Map diseases to doid
    m = set([])
    if mnd1 == mnd2 == 'Disease':
        for r in df.values:
            sources = d2doid[r[0]] if r[0] in d2doid else [r[0]] if r[0].startswith('DOID:') else []
            targets = d2doid[r[1]] if r[1] in d2doid else [r[1]] if r[1].startswith('DOID:') else []
            for s in sources:
                for t in targets:
                    m.add((s,t))

    elif mnd1 == 'Disease':
        for r in df.values:
            sources = d2doid[r[0]] if r[0] in d2doid else [r[0]] if r[0].startswith('DOID:') else []
            targets = [r[1]]
            for s in sources:
                for t in targets:
                    m.add((s,t))

    elif mnd2 == 'Disease':
        for r in df.values:
            sources = [r[0]]
            targets = d2doid[r[1]] if r[1] in d2doid else [r[1]] if r[1].startswith('DOID:') else []
            for s in sources:
                for t in targets:
                    m.add((s,t))

    #--Propagate
    if propagate:
        onto_hits = get_onto_hits(m, onto_dict)
        if not onto_hits['n1'] == onto_hits['n2'] == {}:
            m = propagation(onto_hits, onto_dict,action='propagate') | onto_hits['uncovered']

    return pd.DataFrame(m, columns=['n1','n2'])

def get_inheritable_ontology_dict(node_type,onto_path=onto_path):
    n_otgy = {}
    node_type = node_type.lower().capitalize()
    if node_type in set(os.listdir(onto_path)):
        n_otgy = {}
        for file in os.listdir(onto_path+node_type):
            child2parent = []
            universe = set([])
            otgy = file.split('/')[-1][:-4]
            n_otgy[otgy] = {}

            with open(onto_path+node_type+'/%s'%file,'r') as f:
                f.readline()
                for l in f:
                    h = l.rstrip('\n').split('\t')
                    child2parent.append(h)
                    universe.update(h)
            n_otgy[otgy]['c2p'] = child2parent
            n_otgy[otgy]['universe'] = universe
    return n_otgy

def get_onto_hits(pairs,onto_dict):
    ontology_hits = {'n1':{},'n2':{},'uncovered':set([])}
    covered_pairs = set([])

    m = set(map(tuple,pairs))
    for o in onto_dict:

        uv = onto_dict[o]['universe']
        n1_universe = set([x[0] for x in m]) & uv
        n2_universe = set([x[1] for x in m]) & uv


        r1,r2 = set([]),set([])
        c = 0
        for x in m:

            if x[0] in n1_universe:
                r1.add(x)
                covered_pairs.add(x)
            if x[1] in n2_universe:
                r2.add(x)
                covered_pairs.add(x)

        if len(r1) > 0:
            ontology_hits['n1'][o] =  r1
        if len(r2) > 0:
            ontology_hits['n2'][o] =  r2

    #Getting those pairs that were not mapped into any ontology
    pairs_without_ontology = set(m).difference(covered_pairs)
    ontology_hits['uncovered'].update(pairs_without_ontology)

    return ontology_hits

def propagation(onto_hits,onto_dict,action='propagate'):

    associations = set([])
    #If both nodes have an ontology...
    if len(onto_hits['n1']) > 0 and len(onto_hits['n2']) > 0:

        for n1_otgy in onto_hits['n1']:
            n1_edges = onto_hits['n1'][n1_otgy]
            h1 = onto_dict[n1_otgy]['c2p']

            for n2_otgy in onto_hits['n2']:
                n2_edges = onto_hits['n2'][n2_otgy]
                h2 = onto_dict[n2_otgy]['c2p']
                edges = set(n1_edges) & set(n2_edges)

                if action == 'depropagate':
                    associations.update(DSP.depropagate(edges,h1,h2))
                elif action == 'propagate':
                    associations.update(DSP.propagate(edges,h1,h2))
                else:
                    sys.exit("Invalid propagation action: %s. You must choose between 'depropagate' or 'propagate'"%action)

    #If just the source-nodes have an ontology...
    elif len(onto_hits['n1']) > 0:
        for otgy in onto_hits['n1']:
            h = onto_dict[otgy]['c2p']
            edges = onto_hits['n1'][otgy]

            if action == 'depropagate':
                associations.update(DSP.depropagate(edges,h1=h))
            elif action == 'propagate':
                associations.update(DSP.propagate(edges,h1=h))
            else:
                sys.exit("Invalid propagation action: %s. You must choose between 'depropagate' or 'propagate'"%action)

    #If just the target-nodes have an ontology...
    elif len(onto_hits['n2']) > 0:
        for otgy in onto_hits['n2']:
            h = onto_dict[otgy]['c2p']
            edges = onto_hits['n2'][otgy]

            if action == 'depropagate':
                associations.update(DSP.depropagate(edges,h2=h))
            elif action == 'propagate':
                associations.update(DSP.propagate(edges,h2=h))
            else:
                sys.exit("Invalid propagation action: %s. You must choose between 'depropagate' or 'propagate'"%action)
    return associations

# *************
#   Pipeline
# *************
if __name__ == "__main__":

    #  Directories
    out_path_prop = out_path+'/propagated/'
    if not os.path.exists(out_path_prop):
        os.makedirs(out_path_prop)

    out_path_deprop = out_path+'/depropagated/'
    if not os.path.exists(out_path_deprop):
        os.makedirs(out_path_deprop)

    #Cleaning the output folder
    for out_folder in [out_path_prop,out_path_deprop]:
        for fld in os.listdir(out_folder):
            rmtree(out_folder+fld)

    #Undirected medges
    with open('../../metadata/undirected_metaedges.txt','r') as f:
        undirected_and_same_nodetype_medges =  set(f.read().splitlines())

    # Interate across metaedges
    for metaedge in tqdm(os.listdir(inp_path)):
        if metaedge.count('-') != 2: continue # skipping no metaedge files

        #Creating output folder
        os.mkdir(out_path_prop+metaedge)
        os.mkdir(out_path_deprop+metaedge)

        #Read metaedge triplet
        mnd1 = mps.node2abbr(metaedge.split('-')[0],reverse=True)
        mnd2 = mps.node2abbr(metaedge.split('-')[2],reverse=True)
        e = mps.edge2abbr(metaedge.split('-')[1],reverse=True)
        undirected = metaedge in undirected_and_same_nodetype_medges

        #Get node universes
        universes = {'n1': get_node_universe(mnd1), 'n2':get_node_universe(mnd2)}

        #Get ontology info (if any)
        onto_dict = get_inheritable_ontology_dict(mnd1)
        onto_dict.update(get_inheritable_ontology_dict(mnd2))

        #Iterating across datasets
        files =[inp_path+metaedge+'/%s'%x for x in os.listdir(inp_path+metaedge)]

        for file in tqdm(files, leave=False, desc=metaedge):

            dt_name = file.split('/')[-1].split('.')[0]

            #-- 1) Read file
            df = pd.read_csv(file, sep='\t')[['n1','n2']]

            #-- 2) Remove redundancy
            df = remove_redundancy(df, undirected=undirected)

            #-- 3) Check if there are "propagatable" edges
            if len(onto_dict) > 0:
                onto_hits = get_onto_hits(df.values, onto_dict)

                # If edges can be propagated, do it and rebuilt the df
                if not onto_hits['n1'] == onto_hits['n2'] == {}:
                    df = pd.DataFrame(list(propagation(onto_hits, onto_dict, action='propagate') | onto_hits['uncovered']),
                                      columns=['n1','n2'])

            #-- 4) Map disease to reference vocabulary
            if mnd1 == 'Disease' or mnd2 == 'Disease':
                df = map_diseases(df, mnd1, mnd2, propagate=True)

            #-- 5) Remove redundancy (just in case the mapping/propagation added any)
            df = remove_redundancy(df, undirected=undirected)

            #-- 6) Limit the dataset to the Bioteque node universe
            df = df[(df['n1'].isin(universes['n1'])) & (df['n2'].isin(universes['n2']))]

            #-- 7) Save dataset
            df = df.sort_values(['n1','n2']).reset_index(drop=True) # sort
            df.to_csv(out_path_prop+'/%s/%s.tsv'%(metaedge,dt_name), index=False, sep='\t')

            #-- 8) Save depropagated version
            onto_hits = get_onto_hits(df.values, onto_dict)
            if not onto_hits['n1'] == onto_hits['n2'] == {}:
                df = pd.DataFrame(list(propagation(onto_hits, onto_dict, action='depropagate') | onto_hits['uncovered']),
                                      columns=['n1','n2'])
                df = df.sort_values(['n1','n2']).reset_index(drop=True) # sort
                df.to_csv(out_path_deprop+'/%s/%s.tsv'%(metaedge,dt_name), index=False, sep='\t')

            else:
                #-- If nothing to depropagate just create a symlink to the 'propagate' version to save space
                src_link = out_path_prop+'/%s/%s.tsv'%(metaedge,dt_name)
                trg_link = out_path_deprop+'/%s/%s.tsv'%(metaedge,dt_name)

                if os.path.isfile(trg_link):
                    os.remove(trg_link)
                if os.path.islink(trg_link):
                    os.unlink(trg_link)

                os.symlink(os.path.abspath(src_link), os.path.abspath(trg_link))
