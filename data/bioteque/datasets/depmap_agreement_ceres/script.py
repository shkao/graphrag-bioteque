import sys
import os
import pandas as pd
import numpy as np
import subprocess
import networkx as nx
import uuid
from scipy.spatial.distance import squareform
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps
g2u = mps.get_gene2unip()

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#--Download data
subprocess.Popen('./get_data.sh', shell = True).wait()

# ***************************
#    1)    GEN-cdp-GEN
# ***************************

#-----------Parameters----------------

quantile_cutoff = 0.999 # ~ corresponds to a 0.20 pearson & spearman correlation
min_connected_genes = 0.9 # --> isolated network componets derived from the correlations are skipped after reaching the min_connected_genes
#---------------------------------

sys.stderr.write('Extracting GEN-cdp-GEN interactions\n')

edges = {'pearson':{}, 'spearman':{}}
#cutoffs = {'pearson':pearson_cutoff, 'spearman':spearman_cutoff}

df = pd.read_csv('./CERES_FC.txt', sep='\t', index_col=0).T

for correlation in ['pearson','spearman']:

    sys.stderr.write('Calculating %s pairwise correlations (may take a while)... '%correlation)
    sys.stderr.flush()

    corr = df.corr(correlation)

    #Get index where correlation is higher to a given cutoff
    k = corr.values[np.triu_indices(len(corr), k=1)]
    cutoff = np.quantile(k, quantile_cutoff)
    b = squareform(k > cutoff) # cutoffs[correlation])

    #Getting edges of correlated genes
    rix, cix = np.where(b)
    lbs = np.asarray([x.split()[0] for x in corr.columns])
    for e in zip(lbs[rix],lbs[cix], corr.values[rix,cix]):
        if e[0] not in g2u or e[1] not in g2u: continue
        for up1 in g2u[e[0]]:
            for up2 in g2u[e[1]]:
                if up1 != up2:
                    edges[correlation][tuple(sorted([up1,up2]))] = e[2]

    sys.stderr.write('done!\n')
    sys.stderr.flush()

sys.stderr.write('Inferring edges from the correlations and removing isolated components...\n')
sys.stderr.flush()
#Getting edges from the pearson and spearman consensus. Keep pearson corr as weight
edges = [list(e)+[edges['pearson'][e]] for e in sorted(set(edges['pearson']) & set(edges['spearman']))]

#Removing isolated correlations
g = nx.Graph()
for e in edges:
    g.add_edge(e[0], e[1])

g_uv = set([])
for cmp in sorted(nx.connected_components(g), key=lambda x: len(x), reverse=True):
    g_uv.update(cmp)
    if len(g_uv)/len(g) >= min_connected_genes: break
edges = [e for e in edges if e[0] in g_uv and e[1] in g_uv]

#Writing
with open(out_path+'/GEN-cdp-GEN/%s.tsv'%source, 'w') as o:
    o.write('n1\tn2\tpearson\n')
    for e in edges:
        o.write('%s\t%s\t%.2f\n'%(e[0], e[1], e[2]))


# *********************************
#    2) CLL-bfn-PGN & PGN-pdw-GEN
# *********************************

sys.stderr.write('Extracting CLL-bfn-GEN interactions\n')

#Getting clmapping
cl2id = {}

#--Given mapping
df = pd.read_csv('./cl_models.csv')
cl2id.update(dict(zip(df['model_id'], df['RRID'])))
cl2id.update(dict(zip(df['BROAD_ID'], df['RRID'])))

#--Our mapping
not_mapped = df[pd.isnull(df['RRID'])]
our_mapping = mps.cl2ID(not_mapped['model_name'])
cl2id.update(dict(zip(df['model_id'], our_mapping)))
cl2id.update(dict(zip(df['BROAD_ID'], our_mapping)))

#--Removing ummaped
cl2id = {cl:cl2id[cl] for cl in list(cl2id) if not pd.isnull(cl2id[cl])}

#Getting binary data
df = pd.read_csv('./CERES_binary.tsv', sep ='\t', index_col=0)

#Removing panessential genes
panessentials = set(pd.read_csv('./panessential_tiers.csv', index_col=0)['gene'])
df = df.loc[[g for g in df.index.values if g not in panessentials]]

#Mapping CLs
df.columns = [cl2id[cl] if cl in cl2id else None for cl in df.columns]
df = df[[cl for cl in df.columns if not pd.isnull(cl)]]

#Getting edges (mapping genes on the fly)
g2u = mps.get_gene2unip()
g2depCLs, pg2gns = {}, {}
for cl in df.columns:
    gns = df.index.values[df[cl]==1]

    for g in gns:
        if g in g2u:
            ups = g2u[g]
            for up in ups:
                if up not in g2depCLs:
                    g2depCLs[up] = set([])
                g2depCLs[up].add(cl)

            if len(ups) > 3:
                pert  = '+'.join(sorted(ups)[:3])+'+...'
            else:
                pert = '+'.join(sorted(ups))

            if pert not in pg2gns:
                pg2gns[pert] = set([])

            pg2gns[pert].update(ups)

#Writing
o = {
    'PGN-pdw-GEN':open(out_path+'/PGN-pdw-GEN/%s.tsv'%source,'w'),
    'PGN-bfn-CLL':open(out_path+'/PGN-bfn-CLL/%s.tsv'%source,'w')
    }
for mp in o:
    o[mp].write('n1\tn2\n')

all_pgns = set([])
for pert in pg2gns:
    pert_id =pert+'_DW_'+ str(uuid.uuid4()).split('-')[0]
    all_pgns.add((pert_id,'','KD'))

    dcls = set([])
    for up in pg2gns[pert]:
        o['PGN-pdw-GEN'].write('%s\t%s\n'%(pert_id,up))
        dcls.update(g2depCLs[up])
    for cl in sorted(dcls):
        o['PGN-bfn-CLL'].write('%s\t%s\n'%(pert_id,cl))

#Clossing files
for mp in o:
    o[mp].close()

sys.stderr.write('Done!\n')
