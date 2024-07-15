import sys
import os
import subprocess
sys.path.insert(0, '../../code/kgraph/utils/')
import ontology as ONT
out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#--Download data
subprocess.Popen('./get_cellosaurus.sh', shell = True).wait()

#--Process Ontology
CLS = ONT.CELLOSAURUS('./cellosaurus.txt')

#--Writing
with open(out_path+'/CLL-hsp-CLL/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')

    for r in sorted(CLS.get_child2parent()):
        AC = r[0]
        hit = r[1]
        o.write('%s\t%s\n'%(AC,hit))
