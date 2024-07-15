import os
import sys
import pandas as pd
import subprocess
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps
guv = mps.get_human_reviewed_uniprot()

current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]
out_path = '../../graph/raw/'


#----------Parameters-------------

dorothea_levels = ['C','D']

#---------------------------------

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait()

#Reading
df = pd.read_csv('./omnipath.tsv',sep='\t')

#Keeping only omnipath
df = df[df['dorothea']==True]

#Keeping only human ppis
df = df[(df['ncbi_tax_id_source'] == 9606) & (df['ncbi_tax_id_target'] == 9606)]

#Keeping only proteins in the univrse
df = df[(df['source'].isin(guv)) & (df['target'].isin(guv))]

#Keeping desired confidence levels
levels = set([])
for x in set(df['dorothea_level']):
    for level in dorothea_levels:
        if level in x:
            levels.add(x)
df = df[df['dorothea_level'].isin(levels)]

#--REGULATION
ppis = set(map(tuple,df[['source','target']].values))
with open(out_path+'/GEN-reg-GEN/%s.tsv'%(source),'w') as o:
    o.write('n1\tn2\n')
    for g1,g2 in ppis:
        o.write('%s\t%s\n'%(g1,g2))
        
#--UP-REGULATION
df_upreg = df[df['consensus_stimulation']==True]
ppis = set(map(tuple,df_upreg[['source','target']].values))
with open(out_path+'/GEN-upr-GEN/%s.tsv'%(source),'w') as o:
    o.write('n1\tn2\n')
    for g1,g2 in ppis:
        o.write('%s\t%s\n'%(g1,g2))

#--DOWN-REGULATION
df_dwreg = df[df['consensus_inhibition']==True]
ppis = set(map(tuple,df_dwreg[['source','target']].values))
with open(out_path+'/GEN-dwr-GEN/%s.tsv'%(source),'w') as o:
    o.write('n1\tn2\n')
    for g1,g2 in ppis:
        o.write('%s\t%s\n'%(g1,g2))

sys.stderr.write('Done!\n')
