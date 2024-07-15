#Drug2ATC code from drugbank, kegg and drugcentral
import os
import sys
import collections
import numpy as np
import pandas as pd
import h5py
import xml.etree.ElementTree as ET
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

db_path = './'
mapping_path = '../../metadata/mappings/'
out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

def break_atc(atc):
    return atc[0], atc[:3], atc[:4], atc[:5], atc

inchikey_atc = collections.defaultdict(set)

#-----------
# 1) Kegg
#-----------
sys.stderr.write('Processing KEGG...\n')
kegg2ikey = {}
with open(mapping_path+'/CPD/kegg.tsv', 'r') as f:
    for l in f:
        l = l.rstrip("\n").split("\t")
        if not l[2]: continue
        kegg2ikey[l[0]] = l[2]


with open(db_path+"/br08303.keg", "r") as f:
    for l in f:
        if l[0] == "E":
            atc = l.split()[1]
        if l[0] == "F":
            drug = l.split()[1]
            if drug not in kegg2ikey: continue
            inchikey_atc[kegg2ikey[drug]].update(break_atc(atc))
print(len(inchikey_atc))

#---------------
# 2) Drugcentral
#---------------
sys.stderr.write('Processing Drugcentral...\n')

d2ikey = {}
with open(mapping_path+'/CPD/drugcentral.tsv', 'r') as f:
    for l in f:
        l = l.rstrip("\n").split("\t")
        if not l[1]: continue
        d2ikey[l[0]] = l[1]


#--Reading drug2atc
df = pd.read_csv('drugcentral_atcs.tsv',sep='\t')
for dg, atc in df[['id','code']].values:
    dg = str(dg)
    if dg not in d2ikey: continue
    dg = d2ikey[dg]
    inchikey_atc[dg].update(break_atc(atc))
    
print(len(inchikey_atc))

#--------------
# 3) Drugbank
#--------------
sys.stderr.write('Processing Drugbank...\n')
db2ikey = mps.get_drugbank2ikey()

#--Parsing file
prefix = "{http://www.drugbank.ca}"
tree = ET.parse(db_path+"/full database.xml")

root = tree.getroot()
#-- Getting ATCs
for drug in root:

    # Drugbank ID

    db_id = None
    for child in drug.findall(prefix + "drugbank-id"):
        if "primary" in child.attrib:
            if child.attrib["primary"] == "true":
                db_id = child.text

    if db_id not in db2ikey: continue

    # ATCs
    for atcs in drug.findall(prefix + "atc-codes"):
        for atc in atcs:
            inchikey_atc[db2ikey[db_id]].update(break_atc(atc.attrib["code"]))
print(len(inchikey_atc))

#Writing
with open(out_path+'/CPD-has-PHC/%s.tsv'%source, 'w') as o:
    o.write('n1\tn2\n')
    for dg in sorted(inchikey_atc):
        for atc in sorted(inchikey_atc[dg]):
            o.write('%s\t%s\n'%(dg,atc))

sys.stderr.write('Done!\n')
