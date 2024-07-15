import sys
import os
import numpy as np
sys.path.insert(0,'../../../../code/kgraph/utils/')
import mappers as mps
import ontology as ONT

imp_path = '../../../ontologies/raw_ontologies/'
out_path = '../disease/'

#*********
#1) DOID
#********
source = 'doid'

#getting references
obo_path = imp_path+'doid.obo'
my_ontology = ONT.DOID(path=obo_path)
xrefs = my_ontology.get_xref()

#mapping references
a = set([])
for refs in xrefs.values():
    a.update(refs)
a = list(a) + list(xrefs.keys())
mapping_dict = dict(zip(a,mps.parse_diseaseID(a,skip_unknown=True)))
xrefs = {mapping_dict[x]:xrefs[x] for x in xrefs.keys() if mapping_dict[x] is not None}

#Writing references
diseases = sorted(xrefs)
with open(out_path+'%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')

    for dis in diseases:
        refs = sorted(set([mapping_dict[x] for x in xrefs[dis] if mapping_dict[x] is not None]))

        for ref in refs:
            o.write('%s\t%s\n'%(dis,ref))

#********
#2) HPO
#********
source = 'hpo'

#getting references
obo_path = imp_path+'/hp.obo'
my_ontology = ONT.HPO(path=obo_path)
xrefs = my_ontology.get_xref()

#mapping references
a = set([])
for refs in xrefs.values():
    a.update(refs)
a = list(a) + list(xrefs.keys())
mapping_dict = dict(zip(a,mps.parse_diseaseID(a,skip_unknown=True)))
xrefs = {mapping_dict[x]:xrefs[x] for x in xrefs.keys() if mapping_dict[x] is not None}

#Writing references
diseases = sorted(xrefs)
with open(out_path+'%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')

    for dis in diseases:
        refs = sorted(set([mapping_dict[x] for x in xrefs[dis] if mapping_dict[x] is not None]))

        for ref in refs:
            o.write('%s\t%s\n'%(dis,ref))

#*********************
#3) CTD_MEDIC (MESH)
#*********************
source = 'ctd_medic'

#getting references
obo_path = imp_path+'/CTD_diseases.obo'
my_ontology = ONT.CTD_MEDIC(path=obo_path)
xrefs = my_ontology.get_xref()

#mapping references
a = set([])
for refs in xrefs.values():
    a.update(refs)
a = list(a) + list(xrefs.keys())
mapping_dict = dict(zip(a,mps.parse_diseaseID(a,skip_unknown=True)))
xrefs = {mapping_dict[x]:xrefs[x] for x in xrefs.keys() if mapping_dict[x] is not None}

#Writing references
diseases = sorted(xrefs)
with open(out_path+'%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')

    for dis in diseases:
        refs = sorted(set([mapping_dict[x] for x in xrefs[dis] if mapping_dict[x] is not None]))

        for ref in refs:
            o.write('%s\t%s\n'%(dis,ref))


#********
#4) EFO
#********
source = 'efo'

#getting references
obo_path =  imp_path+'/efo.obo'
my_ontology = ONT.EFO(path=obo_path)
xrefs = my_ontology.get_xref()

#mapping references
a = set([])
for refs in xrefs.values():
    a.update(refs)
a = list(a) + list(xrefs.keys())
mapping_dict = dict(zip(a,mps.parse_diseaseID(a,skip_unknown=True)))
xrefs = {mapping_dict[x]:xrefs[x] for x in xrefs.keys() if mapping_dict[x] is not None}

#Writing references
diseases = sorted(xrefs)
with open(out_path+'%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')

    for dis in diseases:
        refs = sorted(set([mapping_dict[x] for x in xrefs[dis] if mapping_dict[x] is not None]))

        for ref in refs:
            o.write('%s\t%s\n'%(dis,ref))

#************
#5) ORPHANET
#************
source = 'orphanet'

#getting references
obo_path = imp_path+'/ORDO.csv'
my_ontology = ONT.ORPHANET(path=obo_path)
xrefs = my_ontology.get_xref()

#mapping references
a = set([])
for refs in xrefs.values():
    a.update(refs)
a = list(a) + list(xrefs.keys())
mapping_dict = dict(zip(a,mps.parse_diseaseID(a,skip_unknown=True)))
xrefs = {mapping_dict[x]:xrefs[x] for x in xrefs.keys() if mapping_dict[x] is not None}

#Writing references
with open(out_path+'%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')

    for dis in xrefs.keys():
        refs = sorted(set([mapping_dict[x] for x in xrefs[dis] if mapping_dict[x] is not None]))

        for ref in refs:
            o.write('%s\t%s\n'%(dis,ref))

#************
#6) MEDDRA
#************
source = 'meddra'

#getting references
obo_path = imp_path+'/meddra/'
my_ontology = ONT.MEDDRA(path=obo_path)
xrefs = my_ontology.get_meddra2cui()

#mapping references
a = set([])
for refs in xrefs.values():
    a.update(refs)
a = list(a) + list(xrefs.keys())
mapping_dict = dict(zip(a,mps.parse_diseaseID(a,skip_unknown=True)))
xrefs = {mapping_dict[x]:xrefs[x] for x in xrefs.keys() if mapping_dict[x] is not None}

#Writing references
diseases = sorted(xrefs)
with open(out_path+'%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')

    for dis in diseases:
        refs = sorted(set([mapping_dict[x] for x in xrefs[dis] if mapping_dict[x] is not None]))

        for ref in refs:
            o.write('%s\t%s\n'%(dis,ref))

#************
#7) DISGENET
#************
source = 'disgenet'

m = []
a = set([])
# Disgenet mappings were downladed from: https://www.disgenet.org/downloads
obo_path = imp_path+'/disgenet_mappings.tsv'
with open(obo_path,'r') as f:
    f.readline()
    for l in f:
        h = l.rstrip('\n').split('\t')
        umls = 'UMLS:'+h[0]
        vocab = h[2]
        ref = h[3]

        if vocab == 'DO':
            ref = 'DOID:'+ref

        elif vocab == 'MSH':
            ref = 'MESH:'+ref

        elif vocab == 'ORDO':
            ref = 'ORPHA:'+ref

        elif vocab =='HPO':
            pass #They already have "HP"

        else:
            ref = '%s:%s'%(vocab,ref)

        m.append([umls,ref])

m = np.asarray(m,dtype=object)
mapping_dict =dict(zip(m[:,1],mps.parse_diseaseID(m[:,1])))

#Writing references
with open(out_path+'%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')

    for pair in sorted(map(tuple,m)):
        umls = pair[0]
        ref = mapping_dict[pair[1]]
        if ref is None:continue
        o.write('%s\t%s\n'%(umls,ref))

