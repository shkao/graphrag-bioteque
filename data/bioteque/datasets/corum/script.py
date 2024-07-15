import sys
import os
import subprocess
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps
guv = mps.get_human_reviewed_uniprot()

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#--Download the data
subprocess.Popen('./get_data.sh', shell = True).wait()

edges = set([])
with open('./allComplexes.txt', 'r') as f:
    f.readline()
    for l in f:
        r = l.rstrip().split('\t')

        if r[2] != 'Human': continue
        ups = r[5].split(';')
        for g1 in ups:
            for g2 in ups:
                if g1 == g2: continue
                edges.add(tuple(sorted([g1,g2])))

with open(out_path+'/GEN-ppi-GEN/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for i in sorted(edges):
        o.write('%s\t%s\n'%(i[0],i[1]))
sys.stderr.write('Done!\n')

