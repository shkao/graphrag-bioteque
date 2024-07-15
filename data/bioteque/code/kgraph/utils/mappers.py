#Import
import sys
import os
import numpy as np
import copy
import collections
import urllib
from tqdm import tqdm
sys.path.insert(0, '.')
import ontology as ONT
mapping_folder = '/'.join(os.path.realpath(__file__).rstrip('/').split('/')[:-4]) + '/metadata/mappings/'

#General
def get_node2abbr(reverse=False):
    d = {
    'Cell':'CLL',
    'Chemical_entity':'CHE',
    'Cellular_component':'CMP',
    'Compound':'CPD',
    'Disease':'DIS',
    'Domain':'DOM',
    'Gene':'GEN',
    'Molecular_function':'MFN',
    'Pathway':'PWY',
    'Perturbagen':'PGN',
    'Pharmacologic_class':'PHC',
    'Tissue':'TIS'
    }
    if reverse:
        return {y:x for x,y in d.items()}
    else:
        return d

def node2abbr(node,reverse=False):
    d = {x.lower():y for x,y in get_node2abbr(reverse=reverse).items()}
    if type(node)==str:
        node = '_'.join(node.split(' ')).lower()
        if node in d:
            return d[node]
        else:
            return None
    else:
        r = []
        for n in node:
            n = '_'.join(n.split(' ')).lower()
            if n in d:
                r.append(d[n])
            else:
                r.append(None)
        return r

def get_edge2abbr(reverse=False):
    d = {
    'ACETYLATES':'acy',
    'ASSOCIATION':'ass',
    'BAD_FITNESS':'bfn',
    'CAUSES':'cau',
    'CNV_DOWN':'cnd',
    'CNV_UP':'cnu',
    'CODEPENDENCY':'cdp',
    'COEXPRESSION':'cex',
    'COVARIES':'cov',
    'CROSS_REFERENCE':'xrf',
    'CROSSTALK':'ctk',
    'DRUG-DRUG_INTERACTION':'ddi',
    'DEACETYLATES':'dcy',
    'DEMETHYLATES':'dmt',
    'DEPHOSPHORYLATES':'dph',
    'DESUMOYLATES':'dsm',
    'DEUBIQUITINATES':'dub',
    'DEVELOPS_FROM':'dvf',
    'DOWNREGULATES':'dwr',
    'DOWNREGULATION_SENSITIZES':'dws',
    'GOOD_FITNESS':'gfn',
    'HAS_A':'has',
    'HAS_PARENT':'hsp',
    'HAS_PARENT_P':'hsp+',
    'INTERACTS':'int',
    'METHYLATES':'mth',
    'MUTATION':'mut',
    'NEGATIVE_GENETIC_INTERACTION':'ngi',
    'PERTURBS_DOWN':'pdw',
    'PERTURBS_UP':'pup',
    'PHOSPHORYLATES':'pho',
    'POSITIVE_GENETIC_INTERACTION':'pgi',
    'PROTEIN-PROTEIN_INTERACTION':'ppi',
    'PROTEIN_ABUNDANCE':'pab',
    'PROTEIN_DEFICIENCY':'pdf',
    'REGULATES':'reg',
    'RESISTANT':'res',
    'SENSITIVE':'sns',
    'SENSITIZE_UP':'sup',
    'SENSITIZE_DOWN':'sdw',
    'SIMILAR':'sim',
    'SUMOYLATES':'sum',
    'TREATS':'trt',
    'UBIQUITINATES':'ubq',
    'UPREGULATES':'upr',
    'UPREGULATION_SENSITIZES':'ups'
    }
    if reverse:
        return {y:x for x,y in d.items()}
    else:
        return d

def edge2abbr(edge,reverse=False):
    d = {x.lower():y for x,y in get_edge2abbr(reverse=reverse).items()}
    if type(edge)==str:
        edge = '_'.join(edge.split(' ')).lower()
        if edge in d:
            return d[edge]
        else:
            return None
    else:
        r = []
        for e in edge:
            e = '_'.join(e.split(' ')).lower()
            if e in d:
                r.append(d[e])
            else:
                r.append(None)
        return r

def mult_id_iterator(a):
    """
    Used to check different ids inside a string. Each ID in the string must be separated by: ||
    """
    r = a.split('||')
    for i in r:
        yield i

def mapping(array,attribute):
    """
    Calls mapping functions depending on the specified attribute. Attributes that are ambiguous (pathways, proteins...) are not included
    """
    attribute = attribute.lower().capitalize()
    if attribute == 'Cell':
        sys.stderr.write('Mapping CCLs...\n')
        return cl2ID(array)

    elif attribute == 'Tissue':
        sys.stderr.write('Mapping Tissues...\n')
        return tiss2ID(array)

    elif attribute =='Disease':
        sys.stderr.write('Mapping Diseases...\n')
        return parse_diseaseID(array)

    elif attribute == 'Compound':
        sys.stderr.write('Mapping Compounds...\n')
        return drug2ID(array)

    elif attribute == 'Go':
        sys.stderr.write('Mapping Alter GO ids to original ones...\n')
        return go_altID2ID(array)

    else:
        sys.exit('Invalid attribute: %s'%attribute)

#Proteins
def get_human_reviewed_uniprot():
    """Returns a set with the reviewed uniprot human proteins"""
    up_hum_rw = set([])
    with open(mapping_folder+'/GEN/human_reviewed.tsv','r') as f:
        f.readline()
        for l in f:
            up_hum_rw.add(l.split('\t')[0])
    return up_hum_rw

def get_gene2updatedgene():
    g2g = {}
    with open(mapping_folder+'/GEN/gene2updated_gene_HMZ.tsv','r') as f:
        for l in f:
            h = l.rstrip().split('\t')
            g2g[h[0]] = h[1]
    return g2g

def update_genenames(genes):
    """This function uses the mapping file from harmonizome in order to update old gene_names/mouse gene_names to the accepted ones"""

    g2g = get_gene2updatedgene()
    if type(genes) == str:
        if genes in g2g:
            return g2g[genes]
        else:
            return None
    else:
        r  = []
        for g in genes:
            if g in g2g:
                r.append(g2g[g])
            else:
                r.append(None)

        return r

def get_geneID2unip(reviewed=True):

    reviewed_universe = get_human_reviewed_uniprot() if reviewed is True else set([])
    gid2up = {}
    with open(mapping_folder+'/GEN/gid2uniprot.tsv','r') as f:
        for l in f:
            h = l.rstrip('\n').split('\t')
            if reviewed and h[1] not in reviewed_universe: continue
            if h[0] not in gid2up:
                gid2up[h[0]] = set([])
            gid2up[h[0]].add(h[1])

    return gid2up

def get_ensembl2up(reviewed=True, ens_g=True, ens_t=True, ens_p=True):

    reviewed_universe = get_human_reviewed_uniprot() if reviewed is True else set([])
    e2up = {}
    with open(mapping_folder+'/GEN/ens2uniprot.tsv','r') as f:
        for l in f:
            h = l.rstrip('\n').split('\t')
            if reviewed and h[1] not in reviewed_universe: continue
            if not ens_g and h[0].startswith('ENSG'): continue
            if not ens_t and h[0].startswith('ENST'): continue
            if not ens_p and h[0].startswith('ENSP'): continue

            if h[0] not in e2up:
                e2up[h[0]] = set([])
            e2up[h[0]].add(h[1])

    return e2up

def get_gene2unip():
    """Returns a dictionary with gene names as keys and list of uniprot-AC as value from the human reviewed Uniprots"""

    reviewed_universe = get_human_reviewed_uniprot()
    gn2up = {}
    ref_uv = set([])

    with open(mapping_folder+'/GEN/gname2uniprot.tsv','r') as f:
        for l in f:
            h = l.rstrip('\n').split('\t')
            if h[1] not in reviewed_universe: continue
            if h[0] not in gn2up:
                gn2up[h[0]] = set([])
            gn2up[h[0]].add(h[1])
            ref_uv.add(h[0])

    #--Adding synonims if not collpasing with primary names
    with open(mapping_folder+'/GEN/gname_syn2uniprot.tsv','r') as f:
        for l in f:
            h = l.rstrip('\n').split('\t')
            if h[0] in ref_uv: continue
            if h[1] not in reviewed_universe: continue
            if h[0] not in gn2up:
                gn2up[h[0]] = set([])
            gn2up[h[0]].add(h[1])

    return gn2up

def get_up_entry_name2unip():
    import pandas as pd
    """Returns a dictionary with uniprot entry_names as keys and an unique uniprot-AC as value from the human reviewed Uniprots"""

    v = pd.read_csv(mapping_folder+'/GEN/human_reviewed.tsv',sep='\t').values
    return dict(zip(v[:,1], v[:,0]))


def cl2ID(cl):
    """
    Takes a cl name and returns the cellosaurus ID. If the cls is not in the dictionary, it returns None.
    If you want to conver more than one cl is better if you provide a list (much more faster than individual cl)
    """

    #--Reading mapping files
    with open(mapping_folder+'/CLL/cl2id.tsv','r') as f:
        clsaurus = dict([l.rstrip('\n').split('\t') for l in f])
    with open(mapping_folder+'/CLL/cl_syn2id.tsv','r') as f:
        clsaurus2 = dict([l.rstrip('\n').split('\t') for l in f])
    with open(mapping_folder+'/CLL/cl_name_conflicts.txt','r') as f:
        case_conflict_ccls= set(f.read().splitlines())


    if type(cl) == str:
        if cl.startswith('CVCL_'):
            return cl

        if cl not in case_conflict_ccls and cl.upper() not in case_conflict_ccls:
            cl = cl.upper()
        else:
            return None

        #Trying in the original ID
        if cl in clsaurus:
            return clsaurus[cl]

        #Trying in synonyms
        elif cl in clsaurus2:
            return clsaurus2[cl]

        #If nothing works
        else:
            return None

    #If the input is an iterable
    else:
        rs = []
        for the_cl in cl:
            if the_cl.startswith('CVCL_'):
                rs.append(the_cl)
                continue

            if the_cl not in case_conflict_ccls and the_cl.upper() not in case_conflict_ccls:
                the_cl = the_cl.upper()
            else:
                rs.append(None)
                continue
            #Trying in the original ID
            if the_cl in clsaurus:
                rs.append(clsaurus[the_cl])

            #Trying in synonyms
            elif the_cl in clsaurus2:
                rs.append(clsaurus2[the_cl])

            #If nothing works
            else:
                rs.append(None)

        return rs

#Tissues
def get_tiss2bto():

    with open(mapping_folder+'/TIS/tis2id.tsv', 'r') as f:
        tis2bto = dict([l.rstrip('\n').split('\t') for l in f])

    return tis2bto

def tiss2ID(tiss,prioritize_cls=True):
    """
    Automatically checks for multiple ids ('||')
    """
    tiss2bto = get_tiss2bto()

    if type(tiss) == str:
        for hit in mult_id_iterator(tiss):
            if prioritize_cls:
                #First it checks if it exists a CL-id
                r = cl2ID(hit)
                if r is not None:
                    return r
            else:
                if hit.upper() in tiss2bto:
                    return tiss2bto[hit.upper()]

        return None
    else:
        rs = []
        if prioritize_cls:
            #Firts it checks if it exists a CL-id
            r = np.array(cl2ID(tiss))
            nids = np.where(r==None)[0]
            unmapped_tissues = np.array(tiss)[nids]
        else:
            unmapped_tissues = tiss

        for i in unmapped_tissues:
            flag = True
            for hit in mult_id_iterator(i):
                if hit.upper() in tiss2bto:

                    rs.append(tiss2bto[hit.upper()])
                    flag = False
                    break
            if flag:
                rs.append(None)

        if prioritize_cls:
            r[nids] = rs
            return r
        else:
            return rs

def parse_diseaseID(disease,source=None,ignore_sources=set(['MEDGEN','SNOMED','BFO','ICD','NCI']),skip_unknown=False):

    accepted_diseases = ['UMLS, DOID, HP, OMIM, EFO, MESH, ORHPHANET, MEDDRA']

    def get_naked_id(dis):
        symbols = [':','_']
        for s in symbols:
            dis = dis.split(s)[-1].strip()
        dis = dis.split(' ')[0].strip() #This remove string annotations that are following the ID
        return dis

    def parse_source(source,dis,skip_unknown=False):
        if source is None:

            if len(dis) <= 1:
                if skip_unknown is True:
                    return None
                else:
                    sys.exit('Unknown disease: %s. Please specify one of the following sources: \n\n'%dis+', '.join(accepted_diseases))
            elif 'OMIM' in dis.upper():
                source = 'OMIM'
            elif 'ORPHANET' in dis.upper() or 'ORPH' in dis.upper() or 'ORDO' in dis.upper() or 'ORPHA' in dis.upper():
                source = 'ORPHA'
            elif 'DOID' in dis.upper() or 'DO' in dis.upper(): #This must be after ORPHANET to prevent missassociations due to "ORDO"
                source = 'DOID'
            elif 'HP' in dis.upper() or 'HPO' in dis.upper():
                source = 'HP'
            elif 'EFO' in dis.upper():
                source = 'EFO'
            elif 'UMLS' in dis.upper() or 'CUI' in dis.upper() or (len(dis) == 8 and dis.startswith('C')):
                source = 'UMLS'
            elif 'MESH' in dis.upper() or 'MSH' in dis.upper() or (dis[0].isalpha() and dis[1:].isnumeric()):
                source = 'MESH'
            elif 'MEDDRA' in dis.upper() or dis.isnumeric():
                source = 'MEDDRA'
            elif skip_unknown is True:
                soure = None
            else:
                for bad_source in ignore_sources:
                    if bad_source in dis.upper():
                        return None #This returns prevents the following sys exit to happen

                sys.exit('Unknown disease: %s. Please specify one of the following sources: \n\n'%dis+', '.join(accepted_diseases))
        else:
            source = source.upper()

        return source

    #-----------------------------------------------

    #Retriving mappings from alter ids
    if os.path.exists(mapping_folder+'/DIS/doid_alt2id.tsv'):
        with open(mapping_folder+'/DIS/doid_alt2id.tsv', 'r') as f:
            doid_altid2ID = dict([l.rstrip('\n').split('\t') for l in f])
    else:
        doid_altid2ID = {}

    if os.path.exists(mapping_folder+'/DIS/hpo_alt2id.tsv'):
        with open(mapping_folder+'/DIS/hpo_alt2id.tsv', 'r') as f:
            hpo_altid2ID = dict([l.rstrip('\n').split('\t') for l in f])
    else:
        hpo_altid2ID = {}

    if os.path.exists(mapping_folder+'/DIS/meddra_alt2id.tsv'):
        with open(mapping_folder+'/DIS/meddra_alt2id.tsv', 'r') as f:
            meddra_altid2ID = dict([l.rstrip('\n').split('\t') for l in f])
    else:
        meddra_altid2ID = {}

    if type(disease) == str:
        disease = [disease]

    #Iterating through the diseases
    r = []
    for dis in tqdm(disease,desc='Parsing diseases...'):
        the_ontology = parse_source(source,dis,skip_unknown=skip_unknown)

        if the_ontology =='DOID':
            if dis in doid_altid2ID:
                dis = doid_altid2ID[dis]
            naked_id = get_naked_id(dis)
            r.append('DOID:'+naked_id)

        elif the_ontology =='HP' or the_ontology =='HPO':
            if dis in hpo_altid2ID:
                dis = hpo_altid2ID[dis]
            naked_id = get_naked_id(dis)
            r.append('HP:'+naked_id)

        elif the_ontology =='MESH' or the_ontology =='MSH':
            naked_id = get_naked_id(dis)
            r.append('MESH:'+naked_id)

        elif the_ontology == 'OMIM':
            naked_id = get_naked_id(dis)
            r.append('OMIM:'+naked_id)

        elif the_ontology =='EFO':
            naked_id = get_naked_id(dis)
            r.append('EFO:'+naked_id)

        elif the_ontology == 'UMLS':
            naked_id = get_naked_id(dis)
            r.append('UMLS:'+naked_id)

        elif the_ontology =='ORPHANET' or the_ontology == 'ORPHA':
            naked_id = get_naked_id(dis)
            r.append('ORPHA:'+naked_id)

        elif the_ontology == 'MEDDRA':
            if dis in meddra_altid2ID:
                dis = meddra_altid2ID[dis]
            naked_id = get_naked_id(dis)
            r.append('MEDDRA:'+naked_id)
        else:
            r.append(None)

    if len(r) == 1:
        return r[0]
    else:
        return r

#Drugs
def get_drugbank2ikey():
    d = {}
    with open(mapping_folder+'/CPD/drugbank.tsv','r') as f:
        for l in f:
            h = l.rstrip('\n').split('\t')
            d[h[0]] = h[2]
    return d

def get_pharmacodb2ikey():
    d = {}
    with open(mapping_folder+'/CPD/pharmacodb.tsv','r') as f:
        for l in f:
            h = l.rstrip().split('\t')
            d[int(h[0])] = h[1]
    return d

def get_ctd2ikey():
    d = {}
    with open(mapping_folder+'/CPD/ctd.tsv') as f:
        for l in f:
            l = l.rstrip("\n").split("\t")
            if not l[2]: continue
            d[l[0]] = l[2]
    return d

def get_lincs2ikey():
    d = {}
    with open(mapping_folder+'/CPD/lincs.tsv') as f:
        for l in f:
            h = l.rstrip('\n').split('\t')

            if h[2] != '':
                d[h[0]] = h[2]
    return d

def get_nci2ikey():
    d = {}
    with open(mapping_folder+'/CPD/nci60.tsv','r') as f:
        for l in f:
            h = l.rstrip('\n').split('\t')
            if h[2] == '': continue
            d[h[0]] = h[2]
    return d

def get_prism2ikey():
    d = {}
    with open(mapping_folder+'/CPD/prism.tsv','r') as f:
        for l in f:
            h = l.rstrip('\n').split('\t')
            if h[2] == '': continue
            d[h[0]] = h[2]
    return d

def get_sider2ikey():
    d = {}
    with open(mapping_folder+'/CPD/sider.tsv','r') as f:
        for l in f:
            h = l.rstrip('\n').split('\t')
            if h[2] == '': continue
            d[h[0]] = h[2]
    return d

def get_kegg2ikey():
    d = {}
    with open(mapping_folder+'/CPD/kegg.tsv', 'r') as f:
        for l in f:
            l = l.rstrip("\n").split("\t")
            if not l[2]: continue
            d[l[0]] = l[2]
        return d

def drug_name2ID(chem_name,id_type='stdinchikey'):
    """
    It connects to the chemical identifier resolver and maps name to drugID
    """

    authorized_id_types = set(['stdinchikey','smiles'])
    if id_type not in authorized_id_types:
        sys.exit('Invalid id_type: %s'%id_type)

    if type(chem_name) == str:
        for hit in mult_id_iterator(chem_name):
            try:
                hit = urllib.parse.quote(hit)
                return urllib.request.urlopen('https://cactus.nci.nih.gov/chemical/structure/%s/%s'%(hit,id_type)).read().rstrip().decode("utf-8")
            except urllib.error.HTTPError:
                continue

        #If none of the identifiers works...
        return 'None'

    else:
        rs = []
        d2ID = {}
        for d in tqdm(set(chem_name)):
            for hit in mult_id_iterator(d):
                try:
                    hit = urllib.parse.quote(hit)
                    d2ID[hit] = urllib.request.urlopen('https://cactus.nci.nih.gov/chemical/structure/%s/%s'%(hit,id_type)).read().rstrip().decode("utf-8")
                except:
                    continue

        for d in chem_name:
            flag = False
            for hit in mult_id_iterator(d):
                try:
                    hit = urllib.parse.quote(hit)
                    rs.append(urllib.request.urlopen('https://cactus.nci.nih.gov/chemical/structure/%s/%s'%(hit,id_type)).read().rstrip().decode("utf-8"))
                    flag = True
                except urllib.error.HTTPError:
                    continue

            if flag is False:
                rs.append('None')

        return rs
