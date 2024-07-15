import os
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#-------Parameters---------

#Skiping "metabolites" like Copper, Zinc or NADH
skip = set(['RYGMFSIKBFXOCR-UHFFFAOYSA-N', 'DJWUNCQRNNEAKC-UHFFFAOYSA-L', 'HCHKCACWOHOZIP-UHFFFAOYSA-N', 'JIAARYAFYJHUJI-UHFFFAOYSA-L', 'BOPGDPNILDQYTO-NNYOXOHSSA-N'])

#--------------------------

#Retriving universes
gene_universe = mps.get_human_reviewed_uniprot()
db2ikey = mps.get_drugbank2ikey()

all_pairs = {}
active = set([])

#************
# PD targets
#************
label = 'target'
pairs = set([])
with open('./drug_targets.csv','r') as f:
    f.readline()
    for l in f:
        h = l.rstrip('\n').split(',')
        if h[11] != 'Humans':continue
        if h[5] not in gene_universe:continue
        for did in [d.strip() for d in h[12].split(';')]:
            if did not in db2ikey:continue
            pair = (db2ikey[did],h[5])
            pairs.add(pair) 
            
            if pair not in all_pairs:
                all_pairs[pair] = set([])
            all_pairs[pair].add(label)


#--Writing
with open(out_path+'/CPD-int-GEN/%s_pd.tsv'%(source),'w') as o:
    o.write('n1\tn2\n')
    for d,g in sorted(pairs):
        if d in skip: continue
        o.write('%s\t%s\n'%(d,g))

#************
# PK targets
#************
pairs = {}

#--Enzymes    
label = 'enzyme'
with open('./drug_enzymes.csv','r') as f:
    f.readline()
    for l in f:
        h = l.rstrip('\n').split(',')
        if h[11] != 'Humans':continue
        if h[5] not in gene_universe:continue
        for did in [d.strip() for d in h[12].split(';')]:
            if did not in db2ikey:continue
            pair = (db2ikey[did],h[5])
            if pair not in pairs:
                pairs[pair] = set([])
            pairs[pair].add(label)
            
            if pair not in all_pairs:
                all_pairs[pair] = set([])
            all_pairs[pair].add(label)
            
#--Carriers    
label = 'carrier'
with open('./drug_carriers.csv','r') as f:
    f.readline()
    for l in f:
        h = l.rstrip('\n').split(',')
        if h[11] != 'Humans':continue
        if h[5] not in gene_universe:continue
        for did in [d.strip() for d in h[12].split(';')]:
            if did not in db2ikey:continue
            pair = (db2ikey[did],h[5])
            if pair not in pairs:
                pairs[pair] = set([])
            pairs[pair].add(label)
            
            if pair not in all_pairs:
                all_pairs[pair] = set([])
            all_pairs[pair].add(label)

#--Transporters    
label = 'transporter'
with open('./drug_transporters.csv','r') as f:
    f.readline()
    for l in f:
        h = l.rstrip('\n').split(',')
        if h[11] != 'Humans':continue
        if h[5] not in gene_universe:continue
        for did in [d.strip() for d in h[12].split(';')]:
            if did not in db2ikey:continue
            pair = (db2ikey[did],h[5])
            if pair not in pairs:
                pairs[pair] = set([])
            pairs[pair].add(label)
            
            if pair not in all_pairs:
                all_pairs[pair] = set([])
            all_pairs[pair].add(label)
            
#--Writing
with open(out_path+'/CPD-int-GEN/%s_pk.tsv'%(source),'w') as o:
    o.write('n1\tn2\ttype\n')
    for d,g in sorted(pairs):
        if d in skip: continue
        ty = '|'.join(sorted(pairs[(d,g)]))
        o.write('%s\t%s\t%s\n'%(d,g,ty))
            
#****************
# Active targets
#****************
pairs = {}    
 
#--Targets
label = 'target'
with open('./drug_targets_pharmacologically_active.csv','r') as f:
    f.readline()
    for l in f:
        h = l.rstrip('\n').split(',')
        if h[11] != 'Humans':continue
        if h[5] not in gene_universe:continue
        for did in [d.strip() for d in h[12].split(';')]:
            if did not in db2ikey:continue
            pair = (db2ikey[did],h[5])
            if pair not in pairs:
                pairs[pair] = set([])
            pairs[pair].add(label)
            active.add(pair)
            
#--Enzymes    
label = 'enzyme'

with open('./drug_enzymes_pharmacologically_active.csv','r') as f:
    f.readline()
    for l in f:
        h = l.rstrip('\n').split(',')
        if h[11] != 'Humans':continue
        if h[5] not in gene_universe:continue
        for did in [d.strip() for d in h[12].split(';')]:
            if did not in db2ikey:continue
            pair = (db2ikey[did],h[5])
            if pair not in pairs:
                pairs[pair] = set([])
            pairs[pair].add(label)
            active.add(pair)
            
#--Carriers    
label = 'carrier'

with open('./drug_carriers_pharmacologically_active.csv','r') as f:
    f.readline()
    for l in f:
        h = l.rstrip('\n').split(',')
        if h[11] != 'Humans':continue
        if h[5] not in gene_universe:continue
        for did in [d.strip() for d in h[12].split(';')]:
            if did not in db2ikey:continue
            pair = (db2ikey[did],h[5])
            if pair not in pairs:
                pairs[pair] = set([])
            pairs[pair].add(label)
            active.add(pair)            

#--Transporters    
label = 'transporter'

with open('./drug_transporters_pharmacologically_active.csv','r') as f:
    f.readline()
    for l in f:
        h = l.rstrip('\n').split(',')
        if h[11] != 'Humans':continue
        if h[5] not in gene_universe:continue
        for did in [d.strip() for d in h[12].split(';')]:
            if did not in db2ikey:continue
            pair = (db2ikey[did],h[5])
            if pair not in pairs:
                pairs[pair] = set([])
            pairs[pair].add(label)
            active.add(pair)    

#--Writing
with open(out_path+'/CPD-int-GEN/%s_active.tsv'%(source),'w') as o:
    o.write('n1\tn2\ttype\n')
    for d,g in sorted(pairs):
        if d in skip: continue
        ty = '|'.join(sorted(pairs[(d,g)]))
        o.write('%s\t%s\t%s\n'%(d,g,ty))


#ALL drugbank 
with open(out_path+'/CPD-int-GEN/%s.tsv'%(source),'w') as o:
    o.write('n1\tn2\ttype\tis_active\n')
    for d,g in sorted(all_pairs):
        if d in skip: continue
        ty = '|'.join(sorted(all_pairs[(d,g)]))
        if (d,g) in active:
            is_active = 'True'
        else:
            is_active = ''
        o.write('%s\t%s\t%s\t%s\n'%(d,g,ty,is_active))

    sys.stderr.write('Done!\n')

