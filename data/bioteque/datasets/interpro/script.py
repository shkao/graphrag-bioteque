import sys
import os
import subprocess
import gzip
from tqdm import tqdm

#Variables
out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])

source = current_path.split('/')[-1]

d = {}

#--Downloading the data
if not os.path.exists('./9606_reviewed_uniprot2domain.tsv'):
    subprocess.Popen('python get_data.py', shell = True).wait()

#Reading and writing
sys.stderr.write('Reading interpro data...\n')
with open('./9606_reviewed_uniprot2domain.tsv','r') as f, open(out_path+'/GEN-has-DOM/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')

    f.readline()
    for l in f:
        h = l.rstrip('\n').split('\t')
        g = h[0]
        doms = h[1].split('|')
        for dom in sorted(doms):
            o.write('%s\t%s\n'%(g, dom))

sys.stderr.write('Done!\n')
