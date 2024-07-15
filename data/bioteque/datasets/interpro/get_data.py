import sys
import json
import urllib.request, urllib.parse
from tqdm import tqdm
import numpy as np
import pandas as pd
import subprocess
sys.path.insert(0, '../../../utils/')
import mappers as mps

sys.stderr.write('Getting human protein2domain...\n')
sys.stderr.flush()

def get_data(results):
        
        v = []
        for hit in results:
            
            #--Domain
            ipr = hit['metadata']['accession']    
            
            #--Protein
            for p in hit['proteins']:
                v.append([p['accession'].upper(), ipr])
        return v
    
#ALL uniprots we want to retrieve
ups = mps.get_human_reviewed_uniprot()

#Iterating
m = []
error_queries = set([])
for up in tqdm(sorted(ups)):
    
    #Gettinc target_key
    web = 'https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/%s'%up
    
    try:
        handler = urllib.request.urlopen(web)
    except urllib.error.HTTPError: 
        error_queries.add(up)
        continue
        
    r = handler.read().decode( 'utf-8' )
    if r == '': 
        continue
    else:
        r = json.loads(r)
        
    m.extend(get_data(r['results']))
    
    #--In case there ir more than one page
    while r['next'] is not None:
        handler = urllib.request.urlopen(r['next'])
        r = handler.read().decode( 'utf-8' )
        if r == '': break
        r = json.loads(r)
        m.extend(get_data(r['results']))

if len(error_queries) > 0:
    sys.stderr.write("\n\n ***** The following queries couldn't be processed: %s\n\n"%(', '.join(sorted(error_queries))))

#--Writing
m = pd.DataFrame(m, columns=['uniprot', 'domain'])
with open('./9606_reviewed_uniprot2domain.tsv', 'w') as o:
    o.write('uniprot\tdomains\n')
    for up, dta in m.groupby('uniprot'):
        o.write('%s\t%s\n'%(up, '|'.join(np.unique(dta['domain']))))

