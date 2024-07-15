#1) Brenda Tissue Ontology (BTO)
import sys
import os
import subprocess
sys.path.insert(0, '../../code/kgraph/utils/')
import ontology as ONT
out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#--Download data
subprocess.Popen('./get_bto.sh', shell = True).wait()

#--Process Ontology
bto = ONT.BTO('./bto.obo')
human_bto = bto.get_human_bto_set()

#--Writing
with open(out_path+'/TIS-hsp-TIS/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\ttype\n')
    for r in sorted(bto.get_child2parent(get_type=True)):
        t1,t2,typ = r[0],r[1],r[2]
        if t1 in human_bto and t2 in human_bto:
            o.write('%s\t%s\t%s\n'%(t1,t2,typ))
