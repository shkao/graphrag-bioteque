import sys
import pandas as pd
sys.path.insert(0, '../../../utils/')
import mappers as mps
import ontology as ONT
db_path = '/path/to/databases/' # <--- To comply with the dataset licences, this path is not provided

def get_node_universe(node):
    node = node.lower()

    if node == 'gene':
        return gene_universe()
    elif node == 'cell':
        return cell_universe()
    elif node == 'tissue':
        return tissue_universe()
    elif node == 'disease':
        return disease_universe()
    elif node == 'pathway':
        return pathway_universe()
    elif node == 'cellular_component':
        return cellular_component_universe()
    elif node == 'molecular_function':
        return molecular_function_universe()
    elif node == 'domain':
        return domain_universe()
    elif node == 'pharmacologic_class':
        return pharmacologic_class_universe()
    elif node == 'chemical_entity':
        return chemical_entity_universe()

#Gene universe
def gene_universe():
    return mps.get_human_reviewed_uniprot()

#Cell
def cell_universe(p=db_path+'/cellosaurus/'):

    universe = set([])

    #Retriving conflictive cells
    case_conflict_ccls = set([])
    with open(p+'/cellosaurus_name_conflicts.txt','r') as f:
        flag1,flag2 = False,False
        for l in f:

            if l.startswith('Table 3: Cell lines with identical names but different punctuation'):
                flag = False

            if flag1:
                if l == '\n': continue
                case_conflict_ccls.update([x.strip() for x in l.split(':')[0].split(',')])

            if flag2:
                if l == '\n': continue
                case_conflict_ccls.add(l.split(':')[0].rstrip())

            if l.startswith('Table 2: Cell lines with identical names but different casing'):
                flag1 = True
            elif l.startswith('Table 4: Cell lines with a name identical to the synonym of another cell line'):
                flag2 = True

    #Getting universe
    with open(p+'/cellosaurus.txt','r') as f:
         for l in f:
            h = l.rstrip().split('   ')
            hd = h[0]
            if hd == 'AC':
                if h[1] not in case_conflict_ccls:
                    universe.add(h[1])
    return universe

#Tissue
def tissue_universe():
    bto = ONT.BTO()
    universe = bto.get_human_bto_set()
    #Just in case we assure that these terms are not here (they shouldn't as we ask for human bto...)
    for skip_term in ['BTO:0001489','BTO:0000707','BTO:0001143']: #Whole body, Larva, Pupa
        universe.discard(skip_term)
    return universe

#Disease
def disease_universe():
    """
    This disease universe do not have UMLS since it's treated  as an incomplete vocabulary!
    """

    #DOID
    def get_doid_universe():
        doid = ONT.DOID()
        uv = set(doid.get_disease_universe())
        uv.discard('DOID:4') # "Disease"
        return uv

    #HPO
    def get_hpo_universe():
        hpo = ONT.HPO()
        uv = hpo.get_disease_universe()
        uv.discard('HP:0000118') #Phenotypic abnormality
        return uv

    #MESH
    def get_mesh_universe():
        ctd_medic = ONT.CTD_MEDIC()
        uv = ctd_medic.get_mesh_universe()
        uv.discard('MESH:C') #Diseases
        return uv

    #OMIM
    def get_omim_universe(p=db_path+'/diseases/omim/'):
        universe = set([])
        accepted_symbols = set(['NULL','Percent','Number Sign'])
        with open(p+'/mimTitles.txt','r') as f:
            f.readline()
            f.readline()

            for l in f:
                h = l.rstrip('\n').split('\t')

                if h[0] in accepted_symbols:
                    universe.add('OMIM:'+h[1])
        return universe

    #EFO
    def get_efo_universe():
        efo = ONT.EFO()
        uv = set([x for x in efo.get_disease_universe() if 'EFO' in x])
        uv.discard('EFO:0000408') #Disease
        return uv

    #ORPHA
    def get_orphanet_universe():
        orpha = ONT.ORPHANET()
        uv = orpha.get_disease_universe()
        uv.discard('ORPHA:377788') #Disease orpha
        return uv

    #MEDDRA
    def get_meddra_universe():
        meddra = ONT.MEDDRA()
        uv = set(['MEDDRA:'+x for x in meddra.get_id2name()])
        uv.discard('MEDDRA:10022891') # Investigations
        return uv

    #Getting disease universes
    universe = set([])
    universe.update(get_doid_universe()) #DOID
    universe.update(get_hpo_universe()) #HPO
    universe.update(get_mesh_universe()) #MESH
    universe.update(get_omim_universe()) #OMIM
    universe.update(get_efo_universe()) #EFO
    universe.update(get_orphanet_universe()) #ORPHANET
    universe.update(get_meddra_universe()) #MEDDRA

    return universe

#Pathway
def pathway_universe():

    def get_universe_from_cpdb(vocabulary,p=db_path+'/ConsensusPathDB/'):
        universe = set([])
        with open(p+'consensuspathways.tsv','r') as f:
            for l in f:
                h = l.rstrip('\n').split('\t')
                c = h[1].split(':')[-1]
                n = h[0]
                s = h[2]
                if s.lower() == vocabulary.lower():
                    universe.add(c)
        #Add COSMIC mutSIGN
        mutsig = [
        'SBS1', 'SBS2', 'SBS3', 'SBS4', 'SBS5', 'SBS6', 'SBS7a', 'SBS7b',
        'SBS8', 'SBS9', 'SBS10a', 'SBS10b', 'SBS11', 'SBS13', 'SBS14',
        'SBS15', 'SBS17a', 'SBS17b', 'SBS18', 'SBS19', 'SBS20', 'SBS21',
        'SBS22', 'SBS25', 'SBS26', 'SBS28', 'SBS30', 'SBS34', 'SBS35',
        'SBS36', 'SBS37', 'SBS38', 'SBS39', 'SBS40', 'SBSSNP']
        universe.update(mutsig)

        return universe

    #GOBP
    def get_gobp_universe():
        universe = set([])

        with open(db_path+'/go/go-basic.obo','r') as f:
            flag = False
            obsolete = False
            ID = ''

            for l in f:
                if l == '[Term]\n':
                    flag = True
                    ID = ''
                    obsolete = False

                elif l == '\n':

                    #Avoiding obsolete terms
                    if obsolete is False and flag is True:
                        universe.add(ID)

                if flag is True:

                    if 'is_obsolete: true' in l:
                        obsolete = True

                    if l.startswith('id:'):
                        ID = l.rstrip().split('id: ')[-1]

                    elif l.startswith('namespace:'):
                        namespace = l.rstrip().split('namespace: ')[-1]
                        if namespace != 'biological_process':
                            flag = False

        universe.discard('GO:0008150') #Biological process
        return universe

    #Reactome
    def get_reactome_universe():
        return get_universe_from_cpdb('reactome')

    #Kegg
    def get_kegg_universe():
        return get_universe_from_cpdb('kegg')

    #Wikipathways
    def get_wikipathways_universe():
        return get_universe_from_cpdb('wikipathways')

    #Getting pathway universes
    universe = set([])
    universe.update(get_reactome_universe()) #Reactome
    #universe.update(get_gobp_universe()) #GOBP
    #universe.update(get_kegg_universe()) #KEGG
    #universe.update(get_wikipathways_universe()) #Wikipathways
    return universe


#Cellular component
def cellular_component_universe(p=db_path+'/go/'):
    universe = set([])

    with open(p+'/go-basic.obo','r') as f:
        flag = False
        obsolete = False
        ID = ''

        for l in f:
            if l == '[Term]\n':
                flag = True
                ID = ''
                obsolete = False

            elif l == '\n':

                #Avoiding obsolete terms
                if obsolete is False and flag is True:
                    universe.add(ID)

            if flag is True:

                if 'is_obsolete: true' in l:
                    obsolete = True

                if l.startswith('id:'):
                    ID = l.rstrip().split('id: ')[-1]

                elif l.startswith('namespace:'):
                    namespace = l.rstrip().split('namespace: ')[-1]
                    if namespace != 'cellular_component':
                        flag = False
    universe.discard('GO:0005575') #Cellular component
    return universe

#Molecular_function
def molecular_function_universe(p=db_path+'/go/'):
    universe = set([])

    with open(p+'/go-basic.obo','r') as f:
        flag = False
        obsolete = False
        ID = ''

        for l in f:
            if l == '[Term]\n':
                flag = True
                ID = ''
                obsolete = False

            elif l == '\n':

                #Avoiding obsolete terms
                if obsolete is False and flag is True:
                    universe.add(ID)

            if flag is True:

                if 'is_obsolete: true' in l:
                    obsolete = True

                if l.startswith('id:'):
                    ID = l.rstrip().split('id: ')[-1]

                elif l.startswith('namespace:'):
                    namespace = l.rstrip().split('namespace: ')[-1]
                    if namespace != 'molecular_function':
                        flag = False
    universe.discard('GO:0003674')
    return universe

#Domain
def domain_universe(p=db_path+'/interpro/'):
    with open(p+'/9606_domains.tsv','r') as f:
        universe = set([l.split('\t')[0] for l in f])
    return universe

#Chemical_entitiy
def chemical_entity_universe(p=db_path+'/chebi/'):

    universe = set([])

    # #Getting chebis that have inchikey
    # chebi_with_ikey = set([])
    # with open(p+'/ChEBI2ikey.tsv','r') as f:
    #     for l in f:
    #         h = l.rstrip('\n').split('\t')
    #         chebi_with_ikey.add(h[0])

    #Reading chebi universe but skipping those with inchikey
    with open(p+'/chebi.obo.txt','r') as f:
        flag = False
        obsolete = False
        ID = ''

        for l in f:
            if l == '[Term]\n':
                flag = True
                ID = ''
                obsolete = False

            elif l == '\n':

                #Avoiding obsolete terms
                if obsolete is False and flag is True:
                    universe.add(ID)

            if flag is True:

                if 'is_obsolete: true' in l:
                    obsolete = True

                if l.startswith('id:'):
                    ID = l.rstrip().split('id: ')[-1]
                    # #Skipping chebis with ikeys
                    # if ID in chebi_with_ikey:
                    #     flag = False
    return universe

#Pharmacologic_class
def pharmacologic_class_universe(p=db_path+'/ATC/'):

    #--Reading ATC alterations
    atc_alt = pd.read_csv(db_path+'/ATC/atc_alterations.tsv', sep='\t')
    atc_old2new = {}
    for _, data in atc_alt.groupby('Year changed'):
        d = {}
        for x in data.values:
            if x[0] not in d:
                d[x[0]] = set([])
            d[x[0]].add(x[2])

        atc_old2new.update(d)

    universe = set([])
    with open(p+'/ATC.csv','r') as f:
        f.readline()
        for l in f:
            h = l.rstrip('\n').split(',')
            atc = h[0].split('/')[-1].strip()

            atcs = atc_old2new[atc] if atc in atc_old2new else [atc]

            for atc in atcs:
                if not atc.startswith('T'):
                    universe.add(atc)
    return universe
