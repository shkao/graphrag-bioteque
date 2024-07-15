import sys
import os
import subprocess
import gzip
from tqdm import tqdm
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#----------Parameters-------------

min_score = 0.7

#---------------------------------

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait()

#Retriving string mappings
hup = mps.get_human_reviewed_uniprot()
string2up = {}
with open('./string2uniprot.tsv','r') as f:
    f.readline()
    for l in f:
        h = l.rstrip().split('\t')
        st = h[2].split('.')[1]
        up = h[1].split('|')[0]
        if up not in hup: continue
        if st not in string2up:
            string2up[st] = set([])
        string2up[st].add(up)

#Reading the file and writing results
edges = {}
with open('./human_string_links.txt','r') as f:
    f.readline()
    
    for l in tqdm(f):
        h = l.rstrip().split(' ')

        #Checking score
        score = int(h[-1])/1000
        if score < min_score: continue

        #Checking specie
        txid1 = h[0].split('.')[0]
        txid2 = h[1].split('.')[0]
        assert txid1 == txid2 == '9606'

        #Mapping proteins
        p1 = h[0].split('.')[-1]
        p2 = h[1].split('.')[-1]
        if p1 not in string2up or p2 not in string2up: continue
        p1 = string2up[p1]
        p2 = string2up[p2]

        #Kepping interactions
        for g1 in p1:
            for g2 in p2:
                n1,n2 = sorted([g1,g2])
                if (n1,n2) not in edges:
                    edges[(n1,n2)] = []
    
                edges[(n1,n2)].append(score)

#Writing
with open(out_path+'./GEN-ppi-GEN/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\tscore\n')
    for g1,g2 in sorted(edges):
        score = min(edges[(g1,g2)])
        o.write('%s\t%s\t%.3f\n'%(g1,g2,score))

sys.stderr.write('Done!\n')
