#Drug-Drug interactions from the Chemical Checker
import os
import sys
from tqdm import tqdm
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

db2ikey = mps.get_drugbank2ikey()
edges = set([])
with open('./ddi.tsv','r') as f:
    for l in f:
        h = l.rstrip().split('\t')
        if h[0] in db2ikey and h[1] in db2ikey:
            edges.add(tuple(sorted([db2ikey[h[0]], db2ikey[h[1]]])))

#Writing file
with open(out_path+'/CPD-ddi-CPD/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for e in sorted(edges):
        o.write('%s\t%s\n'%(e[0],e[1]))
sys.stderr.write('Done!\n')
