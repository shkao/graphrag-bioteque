import os
import sys
import subprocess
import pandas as pd
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps
guv = mps.get_human_reviewed_uniprot()

current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]
out_path = '../../graph/raw/'

#--Downloading the data
#subprocess.Popen('./get_data.sh', shell = True).wait()

#********
#  PPIs
#********
df = pd.read_csv('./omnipath.tsv',sep='\t')

#Keeping only omnipath
df = df[df['omnipath']==True]

#Keeping only human ppis
df = df[(df['ncbi_tax_id_source'] == 9606) & (df['ncbi_tax_id_target'] == 9606)]

#Keeping only proteins in the univrse
df = df[(df['source'].isin(guv)) & (df['target'].isin(guv))]

#--Getting ppis
ppis = set(map(lambda x: tuple(sorted(x)),df[['source','target']].values))

with open(out_path+'/GEN-ppi-GEN/%s.tsv'%(source),'w') as o:
    o.write('n1\tn2\n')
    for g1,g2 in sorted(ppis):
        o.write('%s\t%s\n'%(g1,g2))

#********
#  PTMs
#********

df = pd.read_csv('./ptms.tsv',sep='\t')

#Keeping only human ppis
df = df[df['ncbi_tax_id']==9606]

#Keeping only proteins in the univrse
df = df[(df['enzyme'].isin(guv)) & (df['substrate'].isin(guv))]

#--Getting phohphorylations
phospho = set(map(tuple,df[['enzyme','substrate']][df['modification']=='phosphorylation'].values))

with open(out_path+'/GEN-pho-GEN/%s.tsv'%(source),'w') as o:
    o.write('n1\tn2\n')
    for g1,g2 in sorted(phospho):
        o.write('%s\t%s\n'%(g1,g2))
        
#--Getting dephosphorylations
dephospho = set(map(tuple,df[['enzyme','substrate']][df['modification']=='dephosphorylation'].values))

with open(out_path+'/GEN-dph-GEN/%s.tsv'%(source),'w') as o:
    o.write('n1\tn2\n')
    for g1,g2 in sorted(dephospho):
        o.write('%s\t%s\n'%(g1,g2))
       
sys.stderr.write('Done!\n')
