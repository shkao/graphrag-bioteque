import os
import sys
import subprocess
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#--------Parameters----------

low_confidence_evidences = set(['IEA','NAS','ND'])

#----------------------------

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait()

human_reviwed = mps.get_human_reviewed_uniprot()
go_dict = {}

with open('./goa_human.gpa','r') as f:
    for _ in range(28):
        f.readline()

    for l in f:
        h = l.rstrip('\n').split('\t')

        uAC = h[1]
        r = h[2]
        go = h[3]
        evidence = h[-1].split('=')[-1]
       
        if uAC not in human_reviwed: continue

        if r =='enables':
            if evidence not in low_confidence_evidences: #Skipping low confidence evidences
                if go not in go_dict: 
                    go_dict[go] = set([])
                go_dict[go].add((uAC,evidence))

#Writing low conf GO
with open(out_path+'/GEN-has-MFN/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\tevidence\n')
    for i,j in go_dict.items():
        for g in j:
            gene = g[0]
            evidence = g[1]
            o.write('%s\t%s\t%s\n'%(gene,i,evidence))          
    
sys.stderr.write('Done!\n')
                        
