# Cell-has-Disease (https://clue.io/cell-app)
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]
#Manual mapping of disease to UMLS (using umls API)
dis2umls = {'bile duct cancer': 'C0740277',
 'bladder cancer': 'C0699885',
 'bone cancer': 'C0279530',
 'brain cancer': 'C0153633',
 'breast cancer': 'C0678222',
 'cervical cancer': 'C0302592',
 'colon cancer': 'C0699790',
 'embryonal cancer': 'C0751364',
 'endometrial cancer': 'C0476089',
 'esophageal cancer': 'C0152018',
 'gastric cancer': 'C0699791',
 'germ cell cancer': 'C0740345',
 'head and neck cancer': 'C3887461',
 'kidney cancer': 'C0740457',
 'leukemia': 'C0023418',
 'liver cancer': 'C0345904',
 'lung cancer': 'C0684249',
 'lymphoma': 'C0024299',
 'myeloma': 'C0026764',
 'neuroblastoma': 'C0027819',
 'ovarian cancer': 'C0029925',
 'pancreatic cancer': 'C0235974',
 'prostate cancer': 'C0600139',
 'skin cancer': 'C0007114',
 'small intestine cancer': 'C0238196',
 'thyroid cancer': 'C0549473'}

#Reading data
m = []

df = pd.read_csv('./Cell_app_export.txt',sep='\t')
for diss, cl in df[[ 'Primary disease', 'Cellosaurus ID']].values:
        if pd.isnull(diss) or diss == '-666':continue 
        
        diss = diss.split(',')
        for dis in diss:
            if dis in dis2umls:   
                m.append([cl,dis2umls[dis]])
#Mapping disease
m = np.asarray(m,dtype=object)
m[:,1] = mps.parse_diseaseID(m[:,1])

#Writing output
with open(out_path+'/CLL-has-DIS/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for i in m:
        o.write('%s\t%s\n'%(i[0],i[1]))
sys.stderr.write('Done!\n')
