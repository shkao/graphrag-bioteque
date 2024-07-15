import os
import sys
import subprocess
import gzip
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#-----------Parameters----------

max_gns = 125 #as Hetionet --> top 125 genes for each CPD-direction-status

#-----------------------------

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait()

#--Mappings
d2i = mps.get_lincs2ikey()
g2u = mps.get_gene2unip()
g2g = mps.get_gene2updatedgene()

#Updating lincs2inchikey with hetionet pert_id info
with open('./dhimmel-lincs-8e9c56a/data/pertinfo/lincs_small_molecules.tsv', 'r') as f:
    f.readline()
    for l in f:
        h = l.rstrip().split('\t')
        if h[-2].startswith('InChIKey='):
            ik = h[-2].split('InChIKey=')[1]
            if h[0] not in d2i:
                d2i[h[0]] = ik
                
#Reading significant up/dwr genes
d2signs = {}
with gzip.open('./dhimmel-lincs-8e9c56a/data/consensi/signif/dysreg-pert_id.tsv.gz','r') as f:
    f.readline()
    for l in f:
        h = l.decode('utf-8').rstrip('\n').split('\t')
        if h[0] not in d2i: continue
        dg = d2i[h[0]]
        if dg not in d2signs:
            d2signs[dg] = {'up_measured':[], 'up_imputed':[], 'down_measured':[], 'down_imputed':[]}
        d2signs[dg]['%s_%s'%(h[5],h[4])].append((float(h[2]), h[3]))
        
dg2s = {}
for dg in d2signs:
    dg2s[dg] = {'up':[], 'down':[]}
    for model, l in d2signs[dg].items():
        
        l = sorted(l, key = lambda x: abs(x[0]), reverse=True)
        dg2s[dg][model.split('_')[0]].extend([x[1] for x in l[:max_gns]])

#Writing
with open(out_path+'/CPD-upr-GEN/%s.tsv'%source, 'w') as ofile_up, open(out_path+'/CPD-dwr-GEN/%s.tsv'%source, 'w') as ofile_dw:
    ofile_up.write('n1\tn2\n')
    ofile_dw.write('n1\tn2\n')
    
    for dg in sorted(dg2s):
        #--up
        up = set([])
        for g in dg2s[dg]['up']:
            if g in g2g:
                g = g2g[g]
            if g in g2u:
                up.update(g2u[g])
        #--dw
        dw = set([])
        for g in dg2s[dg]['down']:
            if g in g2g:
                g = g2g[g]
            if g in g2u:
                dw.update(g2u[g])

        #--Removing incongruencies
        incongruencies = up & dw
        up = up - incongruencies
        dw = dw - incongruencies

        #--Writing
        for g in up:
            ofile_up.write('%s\t%s\n'%(dg, g))
        for g in dw:
            ofile_dw.write('%s\t%s\n'%(dg, g))
            
sys.stderr.write('Done!\n')
