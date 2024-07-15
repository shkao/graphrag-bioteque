import os
import sys
import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
from tqdm import tqdm
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#-------------Parameters----------------

fdr_cutoff = 0.2 #  As suggested by the authors elsewhere in the article (http://msb.embopress.org/content/14/2/e7656)

#---------------------------------------

#Getting mapping
g2up = mps.get_gene2unip()

#Reading file
m = []

df = pd.read_excel('./msb177656-sup-0004-datasetev3.xlsx', skiprows=12, usecols='A:H')
for r in df[df['FDR (IHW)']<fdr_cutoff].values:

    g1 = r[0]
    g2 = r[1]
    pi = float(r[2])
    fdr = float(r[4])

    if g1 in g2up and g2 in g2up:
        for G1 in g2up[g1]:
            for G2 in g2up[g2]:
                if G1 == G2: continue #Removing self-interactions as mainly belong from "epistatic-interactions" which are not interesting when occurring in the same gene
                m.append(sorted([G1,G2])+[pi,fdr])

#Merging duplicated genes  after mapping
new_m = set([])
for gns, v in pd.DataFrame(m, columns=['g1','g2','pi','fdr']).groupby(['g1','g2']):
    if len(v) > 1:
        pi = np.mean(v['pi'])
        fdr = gmean(v['fdr'])
    else:
        pi = v['pi']
        fdr = v['fdr']
    new_m.add(tuple(sorted(gns)+[float(pi),float(fdr)]))

#Writting output
with open(out_path+'/GEN-pgi-GEN/%s.tsv'%source,'w') as pos_o, open(out_path+'/GEN-ngi-GEN/%s.tsv'%source,'w') as neg_o:
    pos_o.write('n1\tn2\tpi_score\tfdr\n')
    neg_o.write('n1\tn2\tpi_score\tfdr\n')
    for r in sorted(new_m):
        if r[2] > 0:
            pos_o.write('%s\t%s\t%.4f\t%.4f\n'%(r[0],r[1],r[2],r[3])) 
        else:
            neg_o.write('%s\t%s\t%.4f\t%.4f\n'%(r[0],r[1],r[2],r[3])) 
sys.stderr.write('Done!\n')
       

