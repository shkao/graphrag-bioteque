import networkx as nx
import sys
db_path = '../../../../metadata/ontologies/raw_ontologies/' # This path is not provided to comply with the dataset licenses

def is_DAG(m):

    G = nx.DiGraph()
    for r in m:
        G.add_edge(r[0],r[1])

    return nx.is_directed_acyclic_graph(G)

class Hierarchy(object):

    def __init__(self, path):
        self.path = path

    def get_all_childs_from_ancestor(self,child2parent,ancestor_list=None,skip_terms=set([]),include_ancestors=False):
        """Given a child2parent ontology and an ancestor it returns all the terms above the ancestor"""

        r = set([])

        if type(ancestor_list) == str:
            ancestor_list = [ancestor_list]
        if type(skip_terms) == str:
            skip_terms = set([skip_terms])
        else:
            skip_terms = set(skip_terms)

        #Recursion function
        def it(i):
            r = set([])
            try:
                childs = parent2child[i]
            except KeyError:
                return r

            for j in childs:
                if j in skip_terms: continue
                r.add(j)
                r.update(it(j)) #recursion

            return r

        #Getting parent2child dict
        parent2child = {}
        for i in child2parent:
            if i[1] not in parent2child:
                parent2child[i[1]] = set([])
            parent2child[i[1]].add(i[0])

        #Checking ancestor list. If it's None, it considers all the ontology
        if ancestor_list is None or len(ancestor_list) ==  0:
            ancestor_list = set(parent2child.keys())

        #Propagating skip_terms from skipped ancestors
        propagated_skip_terms = set([])
        for ancestor in set(ancestor_list) and skip_terms:
            propagated_skip_terms.update(self.get_all_childs_from_ancestor(child2parent,ancestor_list=[ancestor],include_ancestors=True))
        skip_terms.update(propagated_skip_terms)

        #Iterating through the ontology
        for ancestor in ancestor_list:
            if ancestor in skip_terms: continue
            for i in parent2child[ancestor]:
                if i in skip_terms: continue
                r.add(i)
                r.update(it(i))

        #Including acestor if needed
        if include_ancestors is True:
            r.update([x for x in ancestor_list if x not in skip_terms])
        return r


class BTO(Hierarchy):

    def __init__(self, path=None):
        self.path = path if path is not None else db_path+'/brenda_tissue_ontology/bto.obo'

    def get_child2parent(self,get_part_of=True,get_related_to=False,get_develops_from=False,get_type=False):
        child2parent = []
        flag = False
        AC = ''
        H = set([])
        with open(self.path,'r') as f:
            for l in f:
                if l == '[Term]\n':
                    flag = True

                elif l == '\n':
                    if flag:
                        for hit in H:
                            #Keeping results
                            if get_type:
                                child2parent.append([AC,hit[0],hit[1]])
                            else:
                                child2parent.append([AC,hit])

                    #cleaning
                    flag = False
                    AC = ''
                    H = set([])

                if flag is True:

                    if l.startswith('id:'):
                        AC = l.rstrip().split('id: ')[-1]

                    elif l.startswith('is_a:'):
                        h = l.rstrip().split('is_a: ')[-1].split('!')[0].strip()
                        if get_type:
                            h = (h,'is_a')
                        H.add(h)

                    elif get_part_of is True and l.startswith('relationship: part_of '):
                        h = l.rstrip().split('relationship: part_of ')[-1].split('!')[0].strip()
                        if get_type:
                            h = (h,'part_of')
                        H.add(h)

                    elif get_related_to is True and l.startswith('relationship: related_to '):
                        h = l.rstrip().split('relationship: related_to ')[-1].split('!')[0].strip()
                        if get_type:
                            h = (h,'related_to')
                        H.add(h)

                    elif get_develops_from is True and l.startswith('relationship: develops_from '):
                        h = l.rstrip().split('relationship: develops_from ')[-1].split('!')[0].strip()
                        if get_type:
                            h = (h,'develops_from')
                        H.add(h)

        if flag:
            for hit in H:
                #Keeping results
                if get_type:
                    child2parent.append([AC,hit[0],hit[1]])
                else:
                    child2parent.append([AC,hit])

        return child2parent

    def get_human_bto_set(self):
        bto = self.get_child2parent()
        ancestor = 'BTO:0001489' #Whole body
        skip_terms = set(['BTO:0000707','BTO:0001143']) # [Larva, Pupa]
        return self.get_all_childs_from_ancestor(bto,ancestor,skip_terms=skip_terms,include_ancestors=False)

class DOID(Hierarchy):

    def __init__(self, path=None):
        self.path = path if path is not None else db_path+'/diseases/doid/doid.obo'

    def get_disease_universe(self):
        ancestors = ['DOID:4']
        return self.get_all_childs_from_ancestor(self.get_child2parent(),ancestors,include_ancestors=False)

    def get_child2parent(self,is_inferred=False):

        child2parent = []

        flag = False
        AC = ''
        H = set([])
        with open(self.path,'r') as f:
            for l in f:
                if l == '[Term]\n':
                    flag = True

                elif l == '\n':
                    if flag:
                        for hit in H:
                            #Keeping results
                            child2parent.append([AC,hit])

                    #cleaning
                    flag = False
                    AC = ''
                    H = set([])

                if flag is True:

                    if l.startswith('id:'):
                        AC = l.rstrip().split('id: ')[-1]
                    elif l.startswith('is_a:'):
                        h = l.rstrip().split('is_a: ')[-1].split('!')[0].strip()
                        if is_inferred is False and 'is_inferred=' in h:
                            continue
                        H.add(h)

                    elif l == 'is_obsolete: true\n':
                        flag = False
                        AC = ''
                        name = ''

        if flag:
            for hit in H:
                #Keeping results
                child2parent.append([AC,hit])
        return child2parent

    def get_id2name(self,universe=None):
        id2name = {}

        flag = False
        AC = ''
        H = set([])
        with open(self.path,'r') as f:
            for l in f:
                if l == '[Term]\n':
                    flag = True

                elif l == '\n':
                    if flag:
                        if universe is not None and AC not in universe: continue
                        id2name[AC] = name

                    #cleaning
                    flag = False
                    AC = ''
                    H = set([])

                if flag is True:

                    if l.startswith('id:'):
                        AC = l.rstrip().split('id: ')[-1]

                    elif l.startswith('name:'):
                        name = l.rstrip().split('name: ')[-1].strip()

                    elif l == 'is_obsolete: true\n':
                        flag = False
                        AC = ''
                        name = ''

        if flag:
            id2name[AC] = name

        return id2name


    def get_altid2ID(self):
        altid2ID = {}

        flag = False
        AC = ''
        H = set([])
        with open(self.path,'r') as f:
            for l in f:
                if l == '[Term]\n':
                    flag = True

                elif l == '\n':
                    if flag:
                        for hit in H:
                            #Keeping results
                            altid2ID[hit] = AC

                    #cleaning
                    flag = False
                    AC = ''
                    H = set([])

                if flag is True:

                    if l.startswith('id:'):
                        AC = l.rstrip().split('id: ')[-1]
                    elif l.startswith('alt_id:'):
                        h = l.rstrip().split('alt_id: ')[-1].split('!')[0].strip()

                        H.add(h)

                    elif l == 'is_obsolete: true\n':
                        flag = False
                        AC = ''
                        name = ''
        if flag:
            for hit in H:
                #Keeping results
                altid2ID[hit] = AC

        return altid2ID

    def get_xref(self):

        xref = {}

        flag = False
        AC = ''
        H = set([])
        with open(self.path,'r') as f:
            for l in f:
                if l == '[Term]\n':
                    flag = True

                elif l == '\n':
                    if flag:
                        xref[AC] = H


                    #cleaning
                    flag = False
                    AC = ''
                    H = set([])

                if flag is True:

                    if l.startswith('id:'):
                        AC = l.rstrip().split('id: ')[-1]
                    elif l.startswith('xref: '):
                        ID = l.split('xref: ')[-1].rstrip()
                        H.add(ID)
                    elif l == 'is_obsolete: true\n':
                        flag = False
                        AC = ''
                        name = ''
        if flag:
            xref[AC] = H

        return xref

class HPO(Hierarchy):

    def __init__(self, path=None):
        self.path = path if path is not None else db_path+'/hp.obo'

    def get_disease_universe(self):
        ancestors = ['HP:0000118']
        return self.get_all_childs_from_ancestor(self.get_child2parent(),ancestors,include_ancestors=False)

    def get_child2parent(self):

        child2parent = []

        flag = False
        AC = ''
        H = set([])
        with open(self.path,'r') as f:
            for l in f:
                if l == '[Term]\n':
                    flag = True

                elif l == '\n':
                    if flag:
                        for hit in H:
                            #Keeping results
                            child2parent.append([AC,hit])

                    #cleaning
                    flag = False
                    AC = ''
                    H = set([])

                if flag is True:

                    if l.startswith('id:'):
                        AC = l.rstrip().split('id: ')[-1]
                    elif l.startswith('is_a:'):
                        h = l.rstrip().split('is_a: ')[-1].split('!')[0].strip()
                        H.add(h)
                    elif l == 'is_obsolete: true\n':
                        flag = False
                        AC = ''
                        H = set([])
        if flag:
            for hit in H:
                #Keeping results
                child2parent.append([AC,hit])
        return child2parent

    def get_id2name(self,universe=None):
        id2name = {}

        flag = False
        AC = ''
        H = set([])
        with open(self.path,'r') as f:
            for l in f:
                if l == '[Term]\n':
                    flag = True

                elif l == '\n':
                    if flag:
                        if universe is not None and AC not in universe: continue
                        id2name[AC] = name

                    #cleaning
                    flag = False
                    AC = ''
                    H = set([])

                if flag is True:

                    if l.startswith('id:'):
                        AC = l.rstrip().split('id: ')[-1]
                    elif l.startswith('name:'):
                        name = l.rstrip().split('name: ')[-1].strip()
                    elif l == 'is_obsolete: true\n':
                        flag = False
                        AC = ''
                        H = set([])

        if flag:
            id2name[AC] = name

        return id2name

    def get_altid2ID(self):
        altid2ID = {}

        flag = False
        can_be_replaced = False
        AC = ''
        H = set([])
        with open(self.path,'r') as f:
            for l in f:
                if l == '[Term]\n':
                    flag = True

                elif l == '\n':
                    if flag:
                        for hit in H:
                            #Keeping results
                            altid2ID[hit] = AC

                    #cleaning
                    flag = False
                    can_be_replaced = False
                    AC = ''
                    H = set([])

                if flag is True:

                    if l.startswith('id:'):
                        AC = l.rstrip().split('id: ')[-1]
                    elif l.startswith('alt_id:'):
                        h = l.rstrip().split('alt_id: ')[-1].split('!')[0].strip()
                        H.add(h)
                    elif l == 'is_obsolete: true\n':
                        can_be_replaced = True
                        flag = False

                if can_be_replaced is True and l.startswith('replaced_by: '):
                    H.add(AC)
                    AC =  l.rstrip().split('replaced_by: ')[-1].split('!')[0].strip()
                    flag = True

        if flag:
            for hit in H:
                #Keeping results
                altid2ID[hit] = AC

        return altid2ID

    def get_xref(self):

        xref = {}
        flag = False
        AC = ''
        H = set([])
        with open(self.path,'r') as f:
            for l in f:
                if l == '[Term]\n':
                    flag = True

                elif l == '\n':
                    if flag:
                        xref[AC] = H


                    #cleaning
                    flag = False
                    AC = ''
                    H = set([])

                if flag is True:

                    if l.startswith('id:'):
                        AC = l.rstrip().split('id: ')[-1]
                    elif l.startswith('xref: '):
                        ID = l.split('xref: ')[-1].rstrip()
                        H.add(ID)
        if flag:
            xref[AC] = H

        return xref

class CELLOSAURUS(Hierarchy):

    def __init__(self, path=None):
        self.path = path if path is not None else db_path+'/cellosaurus.txt'

    def get_child2parent(self):

        AC = ''
        H = set([])
        child2parent = []
        with open(self.path,'r') as f:

            for i in range(50):
                f.readline()

            #Reading the document
            for l in f:
                h = l.rstrip().split('   ')
                hd = h[0]

                if hd == 'AC':
                    AC = h[1]

                elif hd == 'HI':
                    H.add(h[1].split('!')[0].strip())

                #Making the dict
                elif hd == '//':

                    #ID
                    if AC.startswith('CVCL'):
                        for hit in H:
                            child2parent.append([AC,hit])
                    #Reset
                    AC = ''
                    H = set([])

        return child2parent

class CHEBI(Hierarchy):

    def __init__(self, path):
        self.path = path

        #-- chebi2drug
        chebi2ikey = {}
        flag = False
        ID = ''
        ikey = ''
        obsolete = False

        with open(self.path,'r') as f:
            for l in f:
                if l == '[Term]\n':
                    flag = True
                elif l == '\n' and flag is True:

                    #Avoiding obsolete terms
                    if flag is True:
                        if ikey == '': continue
                        chebi2ikey[ID] = ikey

                    #cleaning
                    flag = True
                    ID = ''
                    obsolete = False
                    ikey = ''

                if flag is True:

                    if 'is_obsolete: true' in l:
                        flag = False
                    if 'property_value: http://purl.obolibrary.org/obo/chebi/inchikey' in l:
                        ikey = l.split('"')[1].strip()
                    if l.startswith('id:'):
                        ID = l.rstrip().split('id: ')[-1]

        self.chb2ikey = chebi2ikey

    def get_child2parent(self,map_chebi2drug=True, get_type=True,allowed_relationships=set([
                                 'has_role',
                                 'is_conjugate_acid_of',
                                 'is_conjugate_base_of',
                                 'is_tautomer_of',
                                 'is_enantiomer_of'])):

        """
        The mapping between ChEBI and ikeys was found parsing the file "ChEBI_complete_3star.sdf"
        in https://www.ebi.ac.uk/chebi/downloadsForward.do.
        The ontology was used to find the hierarchy between ikeys/chebis
        """

        flag = False
        child2parent = []

        with open(self.path,'r') as f:

            for l in f:
                if l == '[Term]\n':
                    flag = True

                elif l == '\n':
                    if flag:
                        if map_chebi2drug and AC in self.chb2ikey:
                            AC = self.chb2ikey[AC]

                        #Iterating through "is_a"
                        for hit in H:
                            if map_chebi2drug and hit in self.chb2ikey:
                                hit = self.chb2ikey[hit]

                            if get_type:
                                child2parent.append([AC,hit,'is_a'])
                            else:
                                 child2parent.append([AC,hit])

                        #Iterating through "relationship"
                        for hits in rel:
                            hit = hits[0]
                            the_type = hits[1]
                            if map_chebi2drug and hit in self.chb2ikey:
                                hit = self.chb2ikey[hit]

                            if get_type:
                                child2parent.append([AC,hit,the_type])
                            else:
                                 child2parent.append([AC,hit])

                    #cleaning
                    flag = False
                    AC = ''
                    H = set([])
                    rel = set([])

                if flag is True:

                    if l.startswith('id:'):
                        AC = l.rstrip().split('id: ')[-1]
                    elif l.startswith('is_a:'):
                        H.add(l.rstrip().split('is_a: ')[-1].split('!')[0].strip())
                    elif l.startswith('relationship:'):
                        h = l.rstrip().split(' ')
                        if h[1] in allowed_relationships:
                            rel.add((h[2],h[1]))

        if flag:
            #Iterating through "is_a"
            for hit in H:
                if get_type:
                    child2parent.append([AC,hit,'is_a'])
                else:
                    child2parent.append([AC,hit])
            #Iterating through "relationship"
            for hits in rel:
                hit = hits[0]
                the_type = hits[1]
                if map_chebi2drug and hit in self.chb2ikey:
                    hit = self.chb2ikey[hit]

                if get_type:
                    child2parent.append([AC,hit,the_type])
                else:
                    child2parent.append([AC,hit])
        return child2parent

class EFO(Hierarchy):

    def __init__(self, path=None):
        self.path = path if path is not None else db_path+'/efo.obo'

    def get_disease_universe(self):
        ancestors = ['EFO:0000408']
        return self.get_all_childs_from_ancestor(self.get_child2parent(),ancestors,include_ancestors=False)
        #return set(np.asarray(self.get_child2parent()).ravel())

    def get_child2parent(self):

        child2parent = []

        flag = False
        AC = ''
        H = set([])
        with open(self.path,'r') as f:
            for l in f:
                if l == '[Term]\n':
                    flag = True

                elif l == '\n':
                    if flag:
                        for hit in H:
                            #Keeping results
                            child2parent.append([AC,hit])

                    #cleaning
                    flag = False
                    AC = ''
                    H = set([])

                if flag is True:

                    if l.startswith('id:'):
                        AC = l.rstrip().split('id: ')[-1]
                        if 'Orphanet' in AC:
                            AC = 'ORPHA:'+AC.split(':')[-1]
                    elif l.startswith('is_a:'):
                        h = l.rstrip().split('is_a: ')[-1].split('!')[0].strip().split()[0]
                        if 'Orphanet' in h:
                            h = 'ORPHA:'+h.split(':')[-1]
                        H.add(h)

                    elif l == 'is_obsolete: true\n':
                        flag = False
                        AC = ''
                        H = set([])
        if flag:
            for hit in H:
                #Keeping results
                child2parent.append([AC,hit])
        return child2parent

    def get_id2name(self,universe=None):

        id2name = {}

        flag = False
        AC = ''
        H = set([])
        with open(self.path,'r') as f:
            for l in f:
                if l == '[Term]\n':
                    flag = True

                elif l == '\n':
                    if flag:
                        if universe is not None and AC not in universe: continue
                        id2name[AC] = name

                    #cleaning
                    flag = False
                    AC = ''
                    H = set([])

                if flag is True:

                    if l.startswith('id:'):
                        AC = l.rstrip().split('id: ')[-1]
                        if 'Orphanet' in AC:
                            AC = 'ORPHA:'+AC.split(':')[-1]
                    elif l.startswith('name:'):
                        name = l.rstrip().split('name: ')[-1].strip()
                        if 'http' in name:
                            name = name.split('{')[0].strip()
                    elif l == 'is_obsolete: true\n':
                        flag = False
                        AC = ''
                        H = set([])

        return id2name

    def get_xref(self):

        xref = {}

        flag = False
        AC = ''
        H = set([])
        with open(self.path,'r') as f:
            for l in f:
                if l == '[Term]\n':
                    flag = True

                elif l == '\n':
                    if flag:
                        xref[AC] = set([x for x in H if x!= AC])

                    #cleaning
                    flag = False
                    AC = ''
                    H = set([])

                if flag is True:

                    if l.startswith('id:'):
                        AC = l.rstrip().split('id: ')[-1]
                        if 'Orphanet' in AC:
                            AC = 'ORPHA:'+AC.split(':')[-1]

                    elif l.startswith('xref: '):

                        ID = l.split('xref: ')[1].split()[0]
                        if 'http' in ID:
                            ID = ID.split('/')[-1]
                        H.add(ID)

                    elif l == 'is_obsolete: true\n':
                        flag = False
                        AC = ''
                        H = set([])

        return xref


class ORPHANET(Hierarchy):
    """By defaul it assumes that the path is a csv (from bioportal)"""

    def __init__(self, path=None,is_csv=None):
        self.path = path if path is not None else db_path+'/ORDO.csv'
        self.is_csv = is_csv if is_csv is not None else True

    def get_disease_universe(self):
        ancestors = ancestors = ['ORPHA:377788','ORPHA:377789','ORPHA:377791','ORPHA:377792','ORPHA:377793']
        return self.get_all_childs_from_ancestor(self.get_child2parent(),ancestors,include_ancestors=True)

    def get_child2parent(self):
        child2parent = []
        if self.is_csv is True:
            import csv
            csv.field_size_limit(10000000) #changing limit

            with open(self.path,'r') as f:
                f = csv.reader(f)
                next(f) #skiping the header
                for h in f:

                    if h[4] == 'True': continue #skipping obsolete terms

                    #Retriving Orph
                    orph = 'ORPHA:' + h[0].split('/')[-1].split('_')[-1].strip()

                    #Retriving parents
                    parent_list = h[7].split('|')
                    parents = set([])
                    for parent in parent_list:
                        parent = 'ORPHA:' + parent.split('/')[-1].split('_')[-1].strip()
                        if parent in set(['','false','owl#Thing','ORPHA:ObsoleteClass']):continue
                        child2parent.append([orph,parent])

        else:
            sys.exit('There is not any implementation for reading a non-CSV MESH file')

        return child2parent

    def get_id2name(self,universe=None):
        id2name = {}
        if self.is_csv is True:
            import csv
            csv.field_size_limit(10000000) #changing limit

            with open(self.path,'r') as f:
                f = csv.reader(f)
                next(f) #skiping the header
                for h in f:

                    if h[4] == 'True': continue #skipping obsolete terms

                    orph = 'ORPHA:' + h[0].split('/')[-1].split('_')[-1].strip() #Retriving orph
                    if universe is not None and orph not in universe: continue
                    name = h[1]  #Retriving name

                    id2name[orph] = name
        else:
            sys.exit('There is not any implementation for reading a non-CSV ORPH file')

        return id2name

    def get_id2namesyn(self,universe=None):
        id2syn = {}
        if self.is_csv is True:
            import csv
            csv.field_size_limit(10000000) #changing limit

            with open(self.path,'r') as f:
                f = csv.reader(f)
                next(f) #skiping the header
                for h in f:

                    if h[4] == 'True': continue #skipping obsolete terms

                    orph = 'ORPHA:' + h[0].split('/')[-1].split('_')[-1].strip() #Retriving orph
                    if universe is not None and orph not in universe: continue
                    synonims = h[2].split('|')  #Retriving name
                    if synonims == ['']: continue

                    if orph not in id2syn:
                        id2syn[orph] = set([])
                    id2syn[orph].update(synonims)
        else:
            sys.exit('There is not any implementation for reading a non-CSV ORPHA file')

        return id2syn

    def get_xref(self,universe=None):
        import numpy as np
        xref = {}
        if self.is_csv is True:
            import csv
            csv.field_size_limit(10000000) #changing limit

            with open(self.path,'r') as f:
                f = csv.reader(f)
                hd = next(f) #skiping the header
                xref_ix = [i for i,x in enumerate(hd) if 'hasDbXref' in x][0]

                for h in f:

                    if h[4] == 'True': continue #skipping obsolete terms

                    orph = 'ORPHA:' + h[0].split('/')[-1].split('_')[-1].strip() #Retriving orph
                    if universe is not None and orph not in universe: continue
                    xrefs = h[xref_ix].split('|')  #Retriving name
                    if xrefs == ['']: continue

                    if orph not in xref:
                        xref[orph] = set([])
                    xref[orph].update(xrefs)
        else:
            sys.exit('There is not any implementation for reading a non-CSV ORPHA file')

        return xref



class CTD_MEDIC(Hierarchy):

    def __init__(self, path=None,is_tsv=None):
        self.path = path if path is not None else db_path+'/CTD_diseases.obo'

    def get_mesh_universe(self):
        universe = set([])

        flag = False
        AC = ''
        H = set([])
        with open(self.path,'r') as f:
            for l in f:
                if l == '[Term]\n':
                    flag = True

                elif l == '\n':
                    if flag:
                        if 'MESH' in AC:
                            universe.add(AC)
                        for hit in H:
                            if 'MESH' in hit:
                                universe.add(hit)

                    #cleaning
                    flag = False
                    AC = ''
                    H = set([])

                if flag is True:

                    if l.startswith('id:'):
                        AC = l.rstrip().split('id: ')[-1]
                    elif l.startswith('is_a:'):
                        h = l.rstrip().split('is_a: ')[-1].split('!')[0].strip()
                        H.add(h)
        if flag:
            for hit in H:
                #Keeping results
                if 'MESH' in hit:
                    universe.add(hit)

        universe.discard('MESH:C') #"Disease"

        return universe

    def get_child2parent(self):
        child2parent = []

        flag = False
        AC = ''
        H = set([])
        with open(self.path,'r') as f:
            for l in f:
                if l == '[Term]\n':
                    flag = True

                elif l == '\n':
                    if flag:
                        for hit in H:
                            #Keeping results
                            child2parent.append([AC,hit])

                    #cleaning
                    flag = False
                    AC = ''
                    H = set([])

                if flag is True:

                    if l.startswith('id:'):
                        AC = l.rstrip().split('id: ')[-1]
                    elif l.startswith('is_a:'):
                        h = l.rstrip().split('is_a: ')[-1].split('!')[0].strip()
                        H.add(h)
        if flag:
            for hit in H:
                #Keeping results
                child2parent.append([AC,hit])

        return child2parent

    def get_id2name(self,universe=None):
        id2name = {}

        flag = False
        AC = ''
        H = set([])
        with open(self.path,'r') as f:
            for l in f:
                if l == '[Term]\n':
                    flag = True

                elif l == '\n':
                    if flag:
                        if universe is None or AC in universe:
                            id2name[AC] = name

                    #cleaning
                    flag = False
                    AC = ''
                    H = set([])

                if flag is True:

                    if l.startswith('id:'):
                        AC = l.rstrip().split('id: ')[-1]
                    elif l.startswith('name:'):
                        name = l.rstrip().split('name: ')[-1].strip()

        if flag:
            if universe is None or AC in universe:
                id2name[AC] = name

        return id2name

    def get_xref(self,universe=None):
        xref = {}

        flag = False
        AC = ''
        H = set([])
        with open(self.path,'r') as f:
            for l in f:
                if l == '[Term]\n':
                    flag = True

                elif l == '\n':
                    if flag:
                        if universe is None or AC in universe:
                            xref[AC] = H

                    #cleaning
                    flag = False
                    AC = ''
                    H = set([])

                if flag is True:

                    if l.startswith('id:'):
                        AC = l.rstrip().split('id: ')[-1]
                    elif l.startswith('alt_id:'):
                        h = l.rstrip().split('alt_id: ')[-1].split('!')[0].strip()
                        if 'DOID' in h:
                            h = 'DOID:'+h.split(':')[-1]
                        H.add(h)
        if flag:
             if universe is None or AC in universe:
                xref[AC] = H


        return xref

    def get_id2namesyn(self,universe=None,exact=True):
        id2syn = {}
        flag = False
        AC = ''
        H = set([])
        with open(self.path,'r') as f:
            for l in f:
                if l == '[Term]\n':
                    flag = True

                elif l == '\n':
                    if flag:
                        if universe is None or AC in universe:
                            id2syn[AC] = H

                    #cleaning
                    flag = False
                    AC = ''
                    H = set([])

                if flag is True:

                    if l.startswith('id:'):
                        AC = l.rstrip().split('id: ')[-1]
                    elif l.startswith('synonym:'):
                        if exact and not l.endswith('EXACT []\n'): continue
                        h = l.rstrip().split('"')[1].strip()

                        H.add(h)
        if flag:
            if universe is None or AC in universe:
                id2syn[AC] = H

        return id2syn

class MEDDRA(Hierarchy):

    def __init__(self, path=None,is_tsv=None):
        self.path = path if path is not None else db_path+'/meddra/'

    def get_child2parent(self):
        child2parent = set([])
        with open(self.path+'/MedDRA/MedAscii/mdhier.asc','r') as f:
            for l in f:
                h = l.rstrip('\n').split('$')
                child2parent.add((h[0],h[1]))
                child2parent.add((h[1],h[2]))
                child2parent.add((h[2],h[3]))

        return list(child2parent)

    def get_altid2ID(self):
        altid2ID = {}
        with open(self.path+'/MedDRA/MedAscii/llt.asc','r') as f:
            for l in f:
                h = l.rstrip('\n').split('$')
                altid2ID[h[0]] = h[2]

        return altid2ID

    def get_id2name(self):
        id2name = {}
        with open(self.path+'/MedDRA/MedAscii/mdhier.asc','r') as f:
            for l in f:
                h = l.rstrip('\n').split('$')

                id2name.update(dict(zip(h[:4],h[4:8])))
        return id2name

    def get_meddra2cui(self,universe=None,skip_SMQ=True,skip_OL=True):
        meddra2cui = {}
        with open(db_path+'/meddraCUIs.tsv','r') as f:
            for l in f:
                h = l.rstrip('\n').split('\t')
                c = h[0]
                t = h[1]
                if skip_SMQ and 'SMQ' in t: continue
                if skip_OL and 'OL' in t: continue
                if universe is not None and md not in universe:continue
                md = h[2]
                if md not in meddra2cui:
                    meddra2cui[md] = set([])
                meddra2cui[md].add(c)
        return meddra2cui


    def get_cui2meddra(self,universe=None,skip_SMQ=True,skip_OL=True):
        altid2ID = self.get_altid2ID()
        # 1) Reading the data
        d = {}
        with open(db_path+'/meddraCUIs.tsv','r') as f:
            for l in f:
                h = l.rstrip('\n').split('\t')
                c = h[0]
                t = h[1]
                md = h[2]
                if universe is not None and md not in univese: continue
                if c not in d:
                    d[c] = set([])
                d[c].add((md,t))


        # 2) Parsing the data
        cui2meddra = {}
        for x,y in d.items():

            flag = False

            #Creating a sub dictionary with all the possible mappings
            hits = {}
            for k in y:
                if k[1] not in hits:
                    hits[k[1]] = set([])
                hits[k[1]].add(k[0])

            #Checking if one of the mappings is already the prefered term (PT)
            if 'PT' in hits:
                cui2meddra[x] = hits['PT']

            #Checking if one of the mappings is a lowest term (LT) that can be mapped to its unique prefered term (PT)
            elif 'LT' in hits:
                cui2meddra[x] = set([altid2ID[i] for i in hits['LT']])

            #Checking if a high level term is available
            elif 'HT' in hits:
                cui2meddra[x] = hits['HT']

            #Checking if a high level group is available
            elif 'HG' in hits:
                cui2meddra[x] = hits['HG']

            #Checking if a system organ class is available
            elif 'OS' in hits:
                cui2meddra[x] = hits['OS']

            #The remainding terms are skiped by default since they are not used in the hierarchy
            else:
                if skip_SMQ is False and 'SMQ' in hits:
                    cui2meddra[x] = hits['SMQ']

                if skip_OL is False and 'OL' in hits:
                    cui2meddra[x] = hits['OL']

                #Asserting that we are covering all the possibilities
                assert len(set(hits).difference(set(['OL','MTH_OL','SMQ','MTH_SMQ']))) == 0


        return cui2meddra

#------------------------------------------------------------------------------------------------------------------------
#    MESH CLASS IS NOT USED AS MESH TERMS ARE OBTAINED FROM CTD_MEDIC (WHICH ALREADY FOCUS ON THE MESH TERMS WE WANT)
#------------------------------------------------------------------------------------------------------------------------

#class MESH(Hierarchy):
#    """By defaul it assumes that the path is a csv (from bioportal)"""

#    def __init__(self, path=None,is_csv=None):
#        self.path = path if path is not None else db_path+'/diseases/mesh/MESH.csv'
#        self.is_csv = is_csv if is_csv is not None else True
#
#    def get_disease_universe(self):
#         ancestors = ['D002318','D064419','D009358','D004066','D007280','D004700','D005128','D005261','D006425','D007154','D052801','D001523',
#         'D009140','D009369','D009422','D009750','D009784','D010038','D010272','D013568','D012140','D017437','D009057','D014777', 'D014947']
#         return self.get_all_childs_from_ancestor(self.get_child2parent(),ancestors,include_ancestors=True)
#
#    def get_child2parent(self):
#        child2parent = []
#        if self.is_csv is True:
#            import csv
#            csv.field_size_limit(10000000) #changing limit
#
#            with open(self.path,'r') as f:
#                f = csv.reader(f)
#                next(f) #skiping the header
#                for h in f:
#
#                    if h[4] == 'True': continue #skipping obsolete terms
#
#                    #Retriving mesh
#                    msh = h[0].split('/')[-1].strip()
#
#                    #Retriving parents
#                    parent_list = h[7].split('|')
#                    parents = set([])
#                    for parent in parent_list:
#                        parent = parent.split('/')[-1].strip()
#                        if parent in set(['','false','owl#Thing']):continue
#                        child2parent.append([msh,parent])
#
#        else:
#            sys.exit('There is not any implementation for reading a non-CSV MESH file')
#
#        return child2parent
#
#    def get_mapped_terms(self,universe=None):
#        id2mapped_terms = {}
#
#        if self.is_csv is True:
#            import csv
#            csv.field_size_limit(10000000) #changing limit
#
#            with open(self.path,'r') as f:
#                f = csv.reader(f)
#                next(f) #skiping the header
#
#                for h in f:
#                    if h[4] == 'True': continue #skipping obsolete terms
#                    ID = h[0].split('/')[-1].strip()
#                    mapps = h[29].split('|')
#
#                    #Skiping if there is not information
#                    if mapps == ['']:continue
#                    if universe is not None and ID not in universe:continue
#
#                    if ID not in id2mapped_terms:
#                        id2mapped_terms[ID] = set([])
#
#                    for mapped_from in mapps:
#                        id2mapped_terms[ID].add(mapped_from.split('/')[-1].strip())
#        else:
#            sys.exit('There is not any implementation for reading a non-CSV MESH file')
#
#        return id2mapped_terms
#
#    def get_id2name(self,universe=None):
#        id2name = {}
#        if self.is_csv is True:
#            import csv
#            csv.field_size_limit(10000000) #changing limit
#
#            with open(self.path,'r') as f:
#                f = csv.reader(f)
#                next(f) #skiping the header
#                for h in f:
#
#                    if h[4] == 'True': continue #skipping obsolete terms
#
#                    msh = h[0].split('/')[-1].strip() #Retriving mesh
#                    if universe is not None and msh not in universe: continue
#                    name = h[1]  #Retriving name
#
#                    id2name[msh] = name
#        else:
#            sys.exit('There is not any implementation for reading a non-CSV MESH file')
#
#        return id2name
#
#    def get_id2namesyn(self,universe=None):
#        id2syn = {}
#        if self.is_csv is True:
#            import csv
#            csv.field_size_limit(10000000) #changing limit
#
#            with open(self.path,'r') as f:
#                f = csv.reader(f)
#                next(f) #skiping the header
#                for h in f:
#
#                    if h[4] == 'True': continue #skipping obsolete terms
#
#                    msh = h[0].split('/')[-1].strip() #Retriving mesh
#                    if universe is not None and msh not in universe: continue
#                    synonims = h[2].split('|')  #Retriving name
#                    if synonims == ['']: continue
#
#                    if msh not in id2syn:
#                        id2syn[msh] = set([])
#                    id2syn[msh].update(synonims)
#        else:
#            sys.exit('There is not any implementation for reading a non-CSV MESH file')
#
#        return id2syn
#
#    def get_mesh2umls(self,universe=None):
#        mesh2umls = {}
#        if self.is_csv is True:
#            import csv
#            csv.field_size_limit(10000000) #changing limit
#
#            with open(self.path,'r') as f:
#                f = csv.reader(f)
#                next(f) #skiping the header
#                for h in f:
#
#                    if h[4] == 'True': continue #skipping obsolete terms
#
#                    msh = h[0].split('/')[-1].strip() #Retriving mesh
#                    if universe is not None and msh not in universe: continue
#                    umls = h[5].split('|')  #Retriving name
#                    if umls == ['']: continue
#
#                    if msh not in mesh2umls:
#                        mesh2umls[msh] = set([])
#                    mesh2umls[msh].update(umls)
#
#        else:
#            sys.exit('There is not any implementation for reading a non-CSV MESH file')
#
#        return mesh2umls
