#Genetic interactions (Eytan:  https://www.nature.com/articles/s41467-018-04647-1)
import os
import sys
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

m = set([])
#Getting mapping
g2up = mps.get_gene2unip() #It already maps tu human reviwed uniprots

#Reading file
with open('./ISLE_network.tsv','r') as f: # <-- Obtained as specified in README.txt
    f.readline()
    for l in f:
        h = l.rstrip().split('\t')

        g1 = h[0]
        g2 = h[1]

        if g1 in g2up and g2 in g2up:
            for G1 in g2up[g1]:
                for G2 in g2up[g2]:
                    if G1 == G2: continue
                    m.add(tuple(sorted([G1,G2])))

#Writting output
with open(out_path+'/GEN-ngi-GEN/%s.tsv'%(source),'w') as o:

    o.write('n1\tn2\ttype\n')
    for l in sorted(m):
        o.write('%s\t%s\tSynthetic_lethal\n'%(l[0],l[1]))
sys.stderr.write('Done!\n')
