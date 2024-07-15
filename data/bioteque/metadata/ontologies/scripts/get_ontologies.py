import sys
import os
import numpy as np
import networkx as nx
import subprocess
out_path = '../inheritable/'
db_path = '../raw_ontologies/'
sys.path.insert(0,'../../../code/kgraph/utils/')
import mappers as mps
import ontology as ONT

def is_DAG(m):

    G = nx.DiGraph()
    for r in m:
        G.add_edge(r[0],r[1])

    return nx.is_directed_acyclic_graph(G)

def write(m,file,header='n1\tn2\n'):
    m = sorted(map(tuple,m))
    if is_DAG(m):
        with open(file,'w') as o:
            if header:
                o.write(header)
            for r in sorted(map(tuple,m)):
                o.write('%s\t%s\n'%(r[0],r[1]))
    else:
        sys.stderr.write('WARNING: %s is NOT a DAG (skipping...\n)')

#*************************
# DOWNLOAD RAW ONTOLOGIES
#*************************
sys.stderr.write('Downloading ontologies...\n')
cmd = 'python3 ../raw_ontologies/get_obos.py'
subprocess.Popen(cmd, shell = True).wait()

#*************************
# INHERITABLE ONTOLOGIES
#*************************
sys.stderr.write('Getting inheritable ontologies...\n')

#Cellular_component

#1) GOCC
obo_path = db_path+'/go-basic.obo'
if not os.path.exists(obo_path):
    sys.stderr.write("Missing file: '%s'.\nSkipping GOCC...\n"%p)
else:
    out = out_path+'/Cellular_component/gocc.tsv'
    M = []
    flag = False
    AC = ''
    H = set([])
    with open(obo_path,'r') as f:

        for l in f:
            if l == '[Term]\n':
                flag = True

            elif l == '\n' and flag is True:

                if AC != '' and H != set([]):
                    for hit in H:
                        if hit.startswith('GO:'):
                            if ns == 'cellular_component':
                                M.append((AC,hit))
                #cleaning
                flag = True
                AC = ''
                H = set([])

            if flag is True:

                if l.startswith('id:'):
                    AC = l.rstrip().split('id: ')[-1]
                elif l.startswith('is_a:'):
                    H.add(l.rstrip().split('is_a: ')[-1].split('!')[0].strip())
                elif l.startswith('namespace:'):
                    ns = l.split('namespace: ')[-1].strip()
    #Writing
    write(M,out)

#Disease

#1) DOID
obo_path = db_path+'doid.obo'
if not os.path.exists(obo_path):
    sys.stderr.write("Missing file: '%s'.\nSkipping DO...\n"%obo_path)
else:
    out = out_path+'/Disease/doid.tsv'
    ont = ONT.DOID(obo_path)

    m = ont.get_child2parent()
    m = np.asarray(m,dtype=object)

    m[:,0] = mps.parse_diseaseID(m[:,0])
    m[:,1] = mps.parse_diseaseID(m[:,1])
    #Writing
    write(m,out)


#2) HPO
obo_path = db_path+'hp.obo'
if not os.path.exists(obo_path):
    sys.stderr.write("Missing file: '%s'.\nSkipping HPO...\n"%obo_path)
else:
    out = out_path+'/Disease/hpo.tsv'
    ont = ONT.HPO(obo_path)

    hpo_universe = ont.get_disease_universe()
    hpo_universe = set(mps.parse_diseaseID(hpo_universe))
    m = [x for x in ont.get_child2parent() if x[0] in hpo_universe and x[1] in hpo_universe]
    m = np.asarray(m,dtype=object)

    m[:,0] = mps.parse_diseaseID(m[:,0])
    m[:,1] = mps.parse_diseaseID(m[:,1])
    #Writing
    write(m,out)

#3) CTD_medic
obo_path = db_path+'CTD_diseases.obo'
if not os.path.exists(obo_path):
    sys.stderr.write("Missing file: '%s'.\nSkipping CTD_medic (MESH)...\n"%obo_path)
else:
    out = out_path+'/Disease/ctd_medic.tsv'
    ont = ONT.CTD_MEDIC(obo_path)

    m = ont.get_child2parent()
    m = np.asarray(m,dtype=object)

    m[:,0] = mps.parse_diseaseID(m[:,0])
    m[:,1] = mps.parse_diseaseID(m[:,1])
    #Writing
    write(m,out)

#4) EFO
obo_path = db_path+'efo.obo'
if not os.path.exists(obo_path):
    sys.stderr.write("Missing file: '%s'.\nSkipping EFO...\n"%obo_path)
else:
    out = out_path+'/Disease/efo.tsv'
    ont = ONT.EFO(obo_path)

    efo_universe = ont.get_disease_universe()
    efo_universe = set(mps.parse_diseaseID(efo_universe))
    m = [x for x in ont.get_child2parent() if x[0] in efo_universe and x[1] in efo_universe]
    m = np.asarray(m,dtype=object)
    #Mapping
    m[:,0] = mps.parse_diseaseID(m[:,0])
    m[:,1] = mps.parse_diseaseID(m[:,1])
    #Writing
    write(m,out)

#5) ORPHANET
obo_path = db_path+'ORDO.csv'
if not os.path.exists(obo_path):
    sys.stderr.write("Missing file: '%s'.\nSkipping Orphanet (ORDO)...\n"%obo_path)
else:

    out = out_path+'/Disease/orphanet.tsv'
    ont = ONT.ORPHANET(obo_path)

    orpha_universe = ont.get_disease_universe()
    orpha_universe = set(mps.parse_diseaseID(orpha_universe))
    m = [x for x in ont.get_child2parent() if x[0] in orpha_universe and x[1] in orpha_universe]
    m = np.asarray(m,dtype=object)
    m[:,0] = mps.parse_diseaseID(m[:,0])
    m[:,1] = mps.parse_diseaseID(m[:,1])
    #Writing
    write(m,out)

#6) MEDDRA
obo_path = db_path+'meddra'
if not os.path.exists(obo_path):
     sys.stderr.write("Missing file: '%s'.\nSkipping MEDDRA...\n"%obo_path)
else:
    out = out_path+'/Disease/meddra.tsv'
    ont = ONT.MEDDRA(obo_path)

    m = ont.get_child2parent()
    m = np.asarray(m,dtype=object)

    m[:,0] = mps.parse_diseaseID(m[:,0])
    m[:,1] = mps.parse_diseaseID(m[:,1])

    #Writing
    write(m,out)

#Domain

#1) INTERPRO
obo_path = db_path+'/interpro_ParentChildTree.txt'
if not os.path.exists(obo_path):
     sys.stderr.write("Missing file: '%s'.\nSkipping Interpro...\n"%obo_path)
else:
    a = []
    with open(obo_path,'r') as f:
        f = f.read().splitlines()
    for l in f:
        h = l.split(':')[0].strip()
        a.append(h.count('-'))
    max_depth = max(a)

    r = {}
    for i in np.arange(2,max_depth+1,2):
        for l in f:
            h = l.split(':')[0].strip()
            if h.count('-') == i-2:
                p = h.split('-')[-1]

                r[p] = set([])
            elif h.count('-') == i:
                r[p].add(h.split('-')[-1])
    m = [[chl,p] for p,childs in r.items() for chl in childs]

    #Writing
    write(m,out)

#Molecular_function

#1) GOMF
obo_path = db_path+'/go-basic.obo'
if not os.path.exists(obo_path):
     sys.stderr.write("Missing file: '%s'.\nSkipping GOMF...\n"%obo_path)
else:
    out = out_path+'/Molecular_function/gomf.tsv'
    M = []
    flag = False
    AC = ''
    H = set([])
    with open(obo_path,'r') as f:

        for l in f:
            if l == '[Term]\n':
                flag = True

            elif l == '\n' and flag is True:

                if AC != '' and H != set([]):
                    for hit in H:
                        if hit.startswith('GO:'):
                            if ns == 'molecular_function':
                                M.append((AC,hit))
                #cleaning
                flag = True
                AC = ''
                H = set([])

            if flag is True:

                if l.startswith('id:'):
                    AC = l.rstrip().split('id: ')[-1]
                elif l.startswith('is_a:'):
                    H.add(l.rstrip().split('is_a: ')[-1].split('!')[0].strip())
                elif l.startswith('namespace:'):
                    ns = l.split('namespace: ')[-1].strip()
    #Writing
    write(M,out)

#Pathway

#1) GOBP
obo_path = db_path+'/go-basic.obo'
if not os.path.exists(obo_path):
     sys.stderr.write("Missing file: '%s'.\nSkipping GOMF...\n"%obo_path)
else:

    out = out_path+'/Pathway/gobp.tsv'
    M = []

    flag = False
    AC = ''
    H = set([])
    with open(obo_path,'r') as f:

        for l in f:
            if l == '[Term]\n':
                flag = True

            elif l == '\n' and flag is True:

                if AC != '' and H != set([]):
                    for hit in H:
                        if hit.startswith('GO:'):
                            if ns == 'biological_process':
                                M.append((AC,hit))
                #cleaning
                flag = True
                AC = ''
                H = set([])

            if flag is True:

                if l.startswith('id:'):
                    AC = l.rstrip().split('id: ')[-1]
                elif l.startswith('is_a:'):
                    H.add(l.rstrip().split('is_a: ')[-1].split('!')[0].strip())
                elif l.startswith('namespace:'):
                    ns = l.split('namespace: ')[-1].strip()

    #Writing
    write(M,out)

#2) Reactome
obo_path = db_path+'/reactome_parent2child.txt'
if not os.path.exists(obo_path):
     sys.stderr.write("Missing file: '%s'.\nSkipping Reactome...\n"%obo_path)
else:
    out = out_path+'/Pathway/reactome.tsv'
    M = []
    for child,parent in pd.read_csv(obo_path, sep='\t')[['Child','Parent']].values:

        if not parent.startswith('R-HSA-') or not child.startswith('R-HSA-'): continue
        M.append((child,parent))
    #Writing
    write(M,out)

#Pharmacologic_class

#1) ATC
obo_path = db_path+'/ATC.csv'
if not os.path.exists(obo_path):
    sys.stderr.write("Missing file: '%s'.\nSkipping ATC...\n"%obo_path)
else:
    out = out_path+'/Pharmacologic_class/atc.tsv'

    df = pd.read_csv(obo_path)
    df = df[df['Obsolete']==False]
    df = df[~pd.isnull(df['Parents'])]

    M  = []
    for atc,parent in df[['Class ID', 'Parents']].values:
        atc = atc.split('/')[-1]
        if len(atc) == 1: continue #Skipping roots
        parent = parent.split('/')[-1]

        M.append((atc,parent)))

    #Writing
    write(M,out)
