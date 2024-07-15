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

#Read the data
gns = mps.get_human_reviewed_uniprot()
s2u = {}
with open('./human.uniprot_2_string.2018.tsv','r') as f:
    for l in f:
        h = l.rstrip().split('\t')
        g = h[1].split('|')[0]
        s = h[2].split('.')[-1]
        if g not in gns: continue
        if s not in s2u:
            s2u[s] = set([])
        s2u[s].add(g)

m = set([])        
with open('./human_compartment_knowledge_full.tsv', 'r') as f:
    
    for l in f:
        h = l.rstrip().split('\t')
        gocc = h[2]
        if gocc == 'GO:0005575': continue # Cellular Component root
        if h[0] in s2u:
            for g in s2u[h[0]]:
                m.add((g,gocc))

with open(out_path+'/GEN-has-CMP/%s.tsv'%source, 'w') as o:
    o.write('n1\tn2\n')
    for r in sorted(m):
        o.write('\t'.join(r)+'\n')

sys.stderr.write('Done!')    
