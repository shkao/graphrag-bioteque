import sys
import os
import subprocess
import urllib.request, urllib.error
import pandas as pd

with open('../scripts/bioontology_apikey.txt','r') as f:
    bioontology_apikey = f.read().splitlines()[0].strip()

#--DO
sys.stderr.write('--> Downloading DO\n')
if os.path.exists('./doid.obo'):
    os.remove('./doid.obo')
cmd = 'wget -O doid.obo https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/master/src/ontology/doid.obo'
subprocess.Popen(cmd, shell = True).wait()

#--HPO
sys.stderr.write('--> Downloading HPO\n')
if os.path.exists('./hp.obo'):
    os.remove('./hp.obo')
cmd = 'wget -O hp.obo https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo'
subprocess.Popen(cmd, shell = True).wait()

#--MESH
sys.stderr.write('--> Downloading MESH (from CTD)\n')
if os.path.exists('./CTD_diseases.obo'):
    os.remove('./CTD_diseases.obo')
cmd = 'wget -O CTD_diseases.obo.gz http://ctdbase.org/reports/CTD_diseases.obo.gz; gunzip --force CTD_diseases.obo.gz'
subprocess.Popen(cmd, shell = True).wait()

#--EFO
sys.stderr.write('--> Downloading EFO\n')
if os.path.exists('./efo.obo'):
    os.remove('./efo.obo')
cmd= 'wget -O efo.obo https://www.ebi.ac.uk/efo/efo.obo'
subprocess.Popen(cmd, shell = True).wait()

#--UMLS (This is NOT an ontology but UMLS-to-OTHERS mappings, accesible through DisGeNET)
sys.stderr.write('--> Downloading UMLS\n')
if os.path.exists('./disease_mappings.tsv'):
    os.remove('./disease_mappings.tsv')
cmd =  'wget https://www.disgenet.org/static/disgenet_ap1/files/downloads/disease_mappings.tsv.gz;\
gunzip --force disease_mappings.tsv.gz;\
mv disease_mappings.tsv disgenet_mappings.tsv'
subprocess.Popen(cmd, shell = True).wait()

#--Orphanet
sys.stderr.write('--> Downloading Orphanet\n')
if os.path.exists('./ORDO.csv'):
    os.remove('./ORDO.csv')

try:
    opener = urllib.request.build_opener()
    opener.addheaders = [('Authorization', 'apikey token=' + bioontology_apikey)]
    url = 'http://data.bioontology.org/ontologies/ORDO/download?download_format=csv'
    a = opener.open(url)
    df = pd.read_csv(a, compression='gzip')
    df.to_csv('./ORDO.csv', index=False)

except urllib.error.HTTPError as e:
    sys.stderr.write('ERROR when downloading ATC --> "%s"\n'%e)
    if str(e) == 'HTTP Error 401: Unauthorized':
        sys.stderr.write('-->Remember to put your bioontology apikey in the first line of the file ../bioontology_apikey.txt\n')


#--MEDDRA (This is NOT the ontology but the MEDDRA-to-UMLS mappings, accessible through SIDER)
sys.stderr.write('--> Downloading Meddra\n')
if os.path.exists('./meddra.tsv'):
    os.remove('./meddra.tsv')
cmd = 'wget http://sideeffects.embl.de/media/download/meddra.tsv.gz; \
gunzip meddra.tsv.gz;\
mv meddra.tsv meddraCUIs.tsv'
subprocess.Popen(cmd, shell = True).wait()

#--GO
sys.stderr.write('--> Downloading GO\n')
if os.path.exists('./go-basic.obo'):
    os.remove('./go-basic.obo')
cmd = 'wget -O go-basic.obo http://purl.obolibrary.org/obo/go/go-basic.obo'
subprocess.Popen(cmd, shell = True).wait()

#--Interpro
sys.stderr.write('--> Downloading Interpro\n')
if os.path.exists('./interpro_ParentChildTree.txt'):
    os.remove('./interpro_ParentChildTree.txt')
cmd = 'wget -O interpro_ParentChildTree.txt ftp://ftp.ebi.ac.uk/pub/databases/interpro/ParentChildTreeFile.txt'
subprocess.Popen(cmd, shell = True).wait()

#--Reactome
sys.stderr.write('--> Downloading Reactome\n')
if os.path.exists('./ReactomePathwaysRelation.txt'):
    os.remove('./ReactomePathwaysRelation.txt')

cmd = 'wget https://reactome.org/download/current/ReactomePathwaysRelation.txt'
subprocess.Popen(cmd,shell=True).wait()

m = []
with open('./ReactomePathwaysRelation.txt','r') as f:
    for l in f:
        h = l.rstrip().split('\t')
        if h[0].startswith('R-HSA') and h[1].startswith('R-HSA'):
            m.append(h)
with open('./parent2child_hsa.txt','w') as o:
    o.write('Parent\tChild\n')
    for r in sorted(m):
        o.write('\t'.join(r)+'\n')

os.remove('./ReactomePathwaysRelation.txt')

#--ATC
sys.stderr.write('--> Downloading ATC\n')
try:
    opener = urllib.request.build_opener()
    opener.addheaders = [('Authorization', 'apikey token=' + bioontology_apikey)]
    url = 'http://data.bioontology.org/ontologies/ATC/download?download_format=csv'
    a = opener.open(url)
    df = pd.read_csv(a, compression='gzip')
    df.to_csv('./ATC.csv', index=False)
except urllib.error.HTTPError as e:
    sys.stderr.write('ERROR when downloading ATC --> "%s"\n'%e)
    if str(e) == 'HTTP Error 401: Unauthorized':
        sys.stderr.write('-->Remember to put your bioontology apikey in the first line of the file ./bioontology_apikey.txt\n')
