import sys
import os
import subprocess
from tqdm import tqdm
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait()

guv = mps.get_human_reviewed_uniprot()

edges = set([])
with open('./intact.txt','r') as f:
    f.readline()
    for l in tqdm(f, desc='Reading Intact data'):
        h = l.rstrip('\n').split('\t')
        if 'taxid:9606' in h[9] and 'taxid:9606' in h[10]:
            g1 = h[0].split('uniprotkb:')[-1]
            g2 = h[1].split('uniprotkb:')[-1]
            if g1 == g2: continue
            if g1 in guv and g2 in guv:
                edges.add(tuple(sorted([g1,g2])))

with open(out_path+'/GEN-ppi-GEN/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for i in edges:
        o.write('%s\t%s\n'%(i[0],i[1]))
sys.stderr.write('Done!\n')
