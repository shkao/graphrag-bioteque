# Cell-has_a-Tissue (https://clue.io/cell-app)
import os
import sys
import pandas as pd
out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

t2id = {'embryonal': 'BTO:0000379',
 'autonomic_ganglia': 'BTO:0002507',
 'biliary_tract': 'BTO:0000122',
 'bone': 'BTO:0000140',
 'breast': 'BTO:0000149',
 'central_nervous_system': 'BTO:0000227',
 'cervix': 'BTO:0001421',
 'colon':'BTO:0000269',
 'endometrium': 'BTO:0001422',
 'fibroblast': 'BTO:0000452',
 'haematopoietic_and_lymphoid_tissue': 'BTO:0000753',
 'head': 'BTO:0000282',
 'kidney': 'BTO:0000671',
 'large_intestine': 'BTO:0000706',
 'liver': 'BTO:0000759',
 'lung': 'BTO:0000763',
 'neck':'BTO:0000420',
 'neuroblastoma': 'BTO:0000931',
 'oesophagus': 'BTO:0000959',
 'ovary': 'BTO:0000975',
 'pancreas': 'BTO:0000988',
 'placenta': 'BTO:0001078',
 'prostate': 'BTO:0001129',
 'salivary_gland': 'BTO:0001203',
 'skin': 'BTO:0001253',
 'small_intestine': 'BTO:0000651',
 'soft_tissue': 'BTO:0001262',
 'stem_cell': 'BTO:0002666',
 'stomach': 'BTO:0001307',
 'thyroid': 'BTO:0001379',
 'urinary_tract': 'BTO:0001244'}

#Reading tiss2CL and mapping tissues
tis2cl = {}
# !! IMPORTANT: You need first to donwload the table "Cell_app_export.txt") from https://clue.io/cell-app

df = pd.read_csv('./Cell_app_export.txt',sep='\t')

for tis, dis, cl in df[['Cell lineage', 'Primary disease', 'Cellosaurus ID']].values:
        if pd.isnull(tis) or tis == '-666':continue 
        
        if ',' in tis:
            ts = [x.strip() for x in tis.split(',')]
        else:
            ts = [tis]
                
        for t in ts: 
            if t == 'engineered': continue
                
            elif t == 'upper_aerodigestive_tract': #There is no term for that so I divided the subcells in this term into their specific tissues
                sub_ts = []
                
                if 'colon' in dis:
                    sub_ts.append('colon')
                if 'esophageal' in dis:
                    sub_ts.append('oesophagus')
                if 'skin' in dis:
                    sub_ts.append('skin')
                if 'head' in dis:
                    sub_ts.append('head')
                if 'neck' in dis:
                    sub_ts.append('neck')
                if 'lung' in dis:
                    sub_ts.append('lung')
                for st in sub_ts:
                    st = t2id[st]
                    if st not in tis2cl:
                        tis2cl[st] = set([])
                    tis2cl[st].add(cl)
            else:
                if t == 'autonomic ganglia':
                    t = 'autonomic_ganglia'
                
                if t not in t2id:
                    continue

                t = t2id[t]
                if t not in tis2cl:
                    tis2cl[t] = set([])
                tis2cl[t].add(cl)

tis2cl = {t:tis2cl[t] for t in list(tis2cl) if len(tis2cl[t])>1}

#Writing output
with open(out_path+'/CLL-has-TIS/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for tis, cls in tis2cl.items():
        for cl in cls:
            o.write('%s\t%s\n'%(cl,tis))
sys.stderr.write('Done!\n')
