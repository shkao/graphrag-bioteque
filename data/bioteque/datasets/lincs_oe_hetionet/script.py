import os
import sys
import subprocess
import bz2
from uuid import uuid4
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#-----------Parameters----------

max_gns = 50 #as Hetionet --> top 50 for each GENE-direction-status

#-----------------------------

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait()

#--Mappings
with open('./dhimmel-lincs-8e9c56a/data/consensi/genes.tsv','r') as f:
    f.readline()
    genes = [x.rstrip().split('\t')[0] for x in f]
g2u = mps.get_geneID2unip()
g2u = {x:g2u[x] for x in genes  if x in g2u}

g2g = mps.get_gene2updatedgene()

#Reading significant up/dwr genes
g2signs = {}
with open('./dhimmel-lincs-8e9c56a/data/consensi/signif/dysreg-overexpression.tsv','r') as f:
    f.readline()
    for l in f:
        h = l.rstrip('\n').split('\t')

        if h[0] not in g2u: continue

        for pert in g2u[h[0]]:
            if pert not in g2signs:
                g2signs[pert] = {'up_measured':[], 'up_imputed':[], 'down_measured':[], 'down_imputed':[]}
            g2signs[pert]['%s_%s'%(h[5],h[4])].append((float(h[2]), h[1]))

g2s = {}
for pert in g2signs:
    g2s[pert] = {'up':[], 'down':[]}
    for model, l in g2signs[pert].items():

        l = sorted(l, key = lambda x: abs(x[0]), reverse=True)
        g2s[pert][model.split('_')[0]].extend([x[1] for x in l[:max_gns]])

#Writing
ofile_up = open(out_path+'/PGN-upr-GEN/%s.tsv'%source, 'w')
ofile_dw = open(out_path+'/PGN-dwr-GEN/%s.tsv'%source, 'w')
ofile_pgn_int = open(out_path+'/PGN-pup-GEN/%s.tsv'%source, 'w')

#--Writing headers
for ofile in [ofile_up, ofile_dw, ofile_pgn_int]:
    ofile.write('n1\tn2\n')

for pert in sorted(g2s):
    tag = str(uuid4()).split('-')[0]

    pert_id = '%s_UP_%s'%(pert, tag)
    #--up
    up = set([])
    for g in g2s[pert]['up']:
        if g in g2u:
            up.update(g2u[g])
    #--dw
    dw = set([])
    for g in g2s[pert]['down']:
        if g in g2u:
            dw.update(g2u[g])

    #--Removing incongruencies
    incongruencies = up & dw
    up = up - incongruencies
    dw = dw - incongruencies

    #--Writing
    for g in up:
        ofile_up.write('%s\t%s\n'%(pert_id, g))
    for g in dw:
        ofile_dw.write('%s\t%s\n'%(pert_id, g))

    if len(up|dw) > 0:
        ofile_pgn_int.write('%s\t%s\n'%(pert_id, pert))

sys.stderr.write('Done!\n')
