import os
import sys
import subprocess
import numpy as np
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

#Variables
out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#-------------Parameters-------------

inferred_sources = set(['HPO','CLINVAR','GWASDB','GWASCAT'])

#------------------------------------

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait()

#Reading
sys.stderr.write('Reading Disgenet data...\n')
m = []
with open('./all_gene_disease_associations.tsv','rb') as f:
    f.readline()

    for l in f:
        h = l.decode(errors='ignore').rstrip('\n').split('\t')

        gid = h[0].strip()
        dis = 'UMLS:'+h[4]
        score = str(round(float(h[9]),3))
        spc = h[2]
        plt = h[3]
        original_sources = [s for s in h[-1].split(';') if s in inferred_sources]

        #Skiping associations that are not considered "curated"
        if len(original_sources) == 0: continue

        try:
            str(round(float(spc),3))
        except:
            pass
        try:
            str(round(float(plt),3))
        except:
            pass

        m.append([gid,dis,score,spc,plt,original_sources])

m = np.asarray(m,dtype=object)

#Mapping
sys.stderr.write('Mapping proteins...\n')
gid2unip = mps.get_geneID2unip()

#Writing
sys.stderr.write('Writing output...\n')
with open(out_path+'/GEN-ass-DIS/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\tscore\tspecificity\tpleiotropy\toriginal_sources\n')
    for r in m:
        gid,dis,s,spc,plt,sr =  str(r[0]),r[1],r[2],r[3],r[4],r[5]
        if gid not in gid2unip: continue
        sr = '|'.join(sr)
        for p in gid2unip[gid]:
            o.write('%s\t%s\t%s\t%s\t%s\t%s\n'%(p,dis,s,spc,plt,sr))

sys.stderr.write('Done!\n')
