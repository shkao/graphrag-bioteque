import sys
import os
import subprocess
import numpy as np
import pandas as pd
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#Renaming tisues to allow the mapping to Brenda ontology
rename_tis = {'cervix, uterine': 'uterine cervix',
             'endometrium 1':'uterine endometrium',
             'endometrium 2':'uterine endometrium',
              'stomach 1':'stomach',
              'stomach 2':'stomach',
              'skin 1': 'skin',
              'skin 2': 'skin',
              'soft tissue 1':'soft body part',
              'soft tissue 2':'soft body part',              
             }

g2g = mps.get_gene2updatedgene()
g2u = mps.get_gene2unip()

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait()

#Reading data
df = pd.read_csv('./normal_tissue.tsv', sep='\t')
df = df[df['Reliability']!='Uncertain']

#Mapping
tis = np.unique([rename_tis[t] if t in rename_tis else t for t in df['Tissue']])
t2id = dict(zip(tis, mps.tiss2ID(tis)))

#Getting protein abudance
tis_pab = set([])
tis_pdf = set([])

for t in tis:
    #If there are tissue replicates I only keep those genes that agree
    if '2' in t: continue
    if '1' in t:
        t1 = t
        t2 = '2'.join(t.split('1'))
        t_id = t2id[rename_tis[t]]
        if t_id is None: continue
        df1 = df[df['Tissue'] == t1]
        df2 = df[df['Tissue'] == t2]
        
        #--Up
        up1 = [g2g[g] if g in g2g else g for g in df1['Gene name'][df1['Level']=='High']]
        up1 = set([u for g in up1 if g in g2u for u in g2u[g]])
        
        up2 = [g2g[g] if g in g2g else g for g in df2['Gene name'][df2['Level']=='High']]
        up2 = set([u for g in up2 if g in g2u for u in g2u[g]])
        up = up1 & up2
        
        #--Down
        dw1 = [g2g[g] if g in g2g else g for g in df1['Gene name'][df1['Level']=='Low']]
        dw1 = set([u for g in up1 if g in g2u for u in g2u[g]])
        
        dw2 = [g2g[g] if g in g2g else g for g in df2['Gene name'][df2['Level']=='Low']]
        dw2 = set([u for g in up2 if g in g2u for u in g2u[g]])
        dw = dw1 & dw2
       
    else:
        df1 = df[df['Tissue']==t]
        if t in rename_tis:
            t = rename_tis[t]
        t_id = t2id[t]
        if t_id is None: continue
        
        #--Up
        up = [g2g[g] if g in g2g else g for g in df1['Gene name'][df1['Level']=='High']]
        up = set([u for g in up if g in g2u for u in g2u[g]])
        
        #--Down
        dw = [g2g[g] if g in g2g else g for g in df1['Gene name'][df1['Level']=='Low']]
        dw = set([u for g in dw if g in g2u for u in g2u[g]])
            
    #Removing incongruencies
    incongruencies = up & dw
    up = up.difference(incongruencies)
    dw = dw.difference(incongruencies)
    
    tis_pab.update(zip([t_id]*len(up), up))
    tis_pdf.update(zip([t_id]*len(dw), dw))
    
#pab
with open(out_path+'/TIS-pab-GEN/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for r in sorted(tis_pab):
        o.write('%s\t%s\n'%(r[0],r[1]))

#pdf
with open(out_path+'/TIS-pdf-GEN/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for r in sorted(tis_pdf):
        o.write('%s\t%s\n'%(r[0],r[1]))

sys.stderr.write('Done!\n')


