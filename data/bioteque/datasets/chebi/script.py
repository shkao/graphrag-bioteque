import os
import sys
import subprocess
sys.path.insert(0, '../../code/kgraph/utils/')
import ontology as ONT

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]
source = 'chebi'

#--Download data
subprocess.Popen('./get_chebi.sh', shell = True).wait()

#--Process Ontology
chebi = ONT.CHEBI('./chebi.obo')
ch2ik = chebi.chb2ikey
res = set([])
for r in sorted(chebi.get_child2parent(map_chebi2drug=False)):
    if r[0] in ch2ik:
        res.add((ch2ik[r[0]], r[1], r[2]))
    if r[1] in ch2ik:
        res.add((ch2ik[r[1]], r[0], r[2]))

#Iterating through chebi and writing file
if not os.path.exists(out_path+'/CPD-has-CHE/'):
    os.mkdir(out_path+'/CPD-has-CHE/')
with open(out_path+'/CPD-has-CHE/%s.tsv'%source, 'w') as o:
    o.write('n1\tn2\ttype\n')
    for r in sorted(res):
        o.write('%s\t%s\t%s\n'%(r[0],r[1],r[2]))

sys.stderr.write('Done!\n')
