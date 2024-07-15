import os
import sys
import subprocess
sys.path.insert(0, '../../code/kgraph/utils/')
from mappers import get_ensembl2up

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait()

#Reading the data
ens2up = get_ensembl2up()
edges = set([])
with open('./HI-union.tsv','r') as f:

    for l in f:
        h = l.rstrip('\n').split('\t')

        if h[0] in ens2up and h[1] in ens2up:
            for g1 in ens2up[h[0]]:
                for g2 in ens2up[h[1]]:
                    if g1 == g2: continue
                    edges.add(tuple(sorted([g1,g2])))
edges = sorted(edges)

if not os.path.exists(out_path+'/GEN-ppi-GEN/'):
    os.mkdir(out_path+'/GEN-ppi-GEN/')

with open(out_path+'/GEN-ppi-GEN/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for i in edges:
        o.write('%s\t%s\n'%(i[0],i[1]))
sys.stderr.write('Done!\n')
