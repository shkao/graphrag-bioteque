import os
import sys
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps
out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#-----------Parameters-----------

#--Cutoffs. The second one is a relaxed cutoff in case the first one gives NO disease for a given drug
cutoffs = [0.05, 0.05]
placebo_margin = [0.05, 0]

#--------------
            
def get_dis_from_drug(drug_df, cutoff, placebo_margin,
                     good_tags = set(['common', 'very frequent', 'frequent' ,'very common'])):
    v = []

    for dis, dta in drug_df.groupby('umls'):
        if 'real' not in set(dta['placebo']): continue

        mn_real = np.median(dta[dta['placebo']=='real']['lbound'])
        mx_real = np.median(dta[dta['placebo']=='real']['ubound'])
        mean = np.mean([mn_real, mx_real])
        
        #If its a well know side effect just keep it
        good_names = set(dta['freq. desc.'][dta['placebo']=='real']) & good_tags
        if len(good_names) > 0:
            v.append([dis, '|'.join(list(good_names)), ''])
            continue

        #Else check if the disease is enough represented in the population 
        if mean >= cutoff:
            
            desc = '%i-%i%%'%(mn_real*100, mx_real*100)

            #--If placebo
            if 'placebo' in set(dta['placebo']):
                mn_placebo = np.median(dta[dta['placebo']=='placebo']['lbound'])
                mx_placebo = np.median(dta[dta['placebo']=='placebo']['ubound'])
                mean_placebo = np.mean([mn_placebo, mx_placebo])

                if mean > (placebo_margin + mean_placebo):
                    v.append([dis, desc, '%i-%i%%'%(mn_placebo*100, mx_placebo*100)])

            #--If NO placebo
            else:
                v.append([dis, desc, ''])
    return v

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait()

#--Reading data
d2i = mps.get_sider2ikey()          
df = pd.read_csv('./meddra_freq.tsv',sep='\t', header=None, names = ['stitch flat', 'stitch stereo', 'umls', 'placebo', 'freq. desc.', 'lbound', 'ubound', 'meddra_hyerarchy', 'meddra id', 'name'])

#Removing uncommon stuff
df = df[~df['freq. desc.'].isin(['very rare', 'uncommon', 'infrequent', 'rare'])]

#Defining placebo an real
df['placebo'] = list(df['placebo'].fillna('real'))

#Getting relevant CPD-DIS associations
m = []
for dg, drug_df in tqdm(df.groupby(['stitch stereo'])):
        
    drug_has_dis = False
    
    #First attemp
    v = get_dis_from_drug(drug_df, cutoffs[0], placebo_margin[0])
    
    if len(v) == 0:
        #Second attemp with relaxed cutoffs
        v = get_dis_from_drug(drug_df, cutoffs[1], placebo_margin[1])
    
    m.extend([[dg]+x for x in v])
        
df = pd.DataFrame(m, columns=['cpd','dis', 'desc', 'placebo'])

#Mapping compounds
df['cpd'] = [d2i[c] if c in d2i else None for c in df['cpd']]
df = df[~pd.isnull(df['cpd'])]

#Mapping diseases
df['dis'] = mps.mapping(np.asarray(df['dis']),'Disease')

#Mapping drugs and writing
with open(out_path+'/CPD-cau-DIS/%s.tsv'%source, 'w')  as o:
    o.write('n1\tn2\tdesc\tplacebo\n')
    for r in df.sort_values(['cpd','dis']).values:
        o.write('\t'.join(r)+'\n')

sys.stderr.write('Done!\n')
