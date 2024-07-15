import os
import sys
import subprocess
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait()

gns = mps.get_human_reviewed_uniprot()

edges = set([])
with open('./UniProt2Reactome_All_Levels.txt','r') as f:
    for l in f:
        h = l.rstrip('\n').split('\t')
        if not h[1].startswith('R-HSA-'): continue
        if h[0] not in gns: continue
        up,rc = h[0],h[1]
        edges.add((h[0], h[1]))
        
with open(out_path+'/GEN-ass-PWY/reactome.tsv','w') as o:
    o.write('n1\tn2\n')
    for r in sorted(edges, key=lambda x: x[1]):
        o.write('\t'.join(r)+'\n')
        
sys.stderr.write('Done\n')

