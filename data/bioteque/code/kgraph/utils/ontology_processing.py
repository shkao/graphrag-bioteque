import numpy as np
import pandas as pd
from tqdm import tqdm

def get_c2p_and_p2c(edges, is_parent2child=False, include_itself=True, header=False):
    """
    edges --> [[child1, parent1], [child1, parent2], [child2, parent1]]
    is_parent2child --> if True then edges  = [[parent1, child1],[parent1, child2],...]
    include_itself --> if True it includes the key in the values
    header --> If true it will skip the first row of the edges (only works if edges is a path)
    """
    if type(edges) == str:
        sep = ',' if edges.endswith('csv') else '\t' if edges.endswith('tsv') else ' '
        header = None if header is False else 0
        edges = pd.read_csv(edges,sep=sep, header=header).values

    # First it gets the mappings from the edges
    c2p,p2c = {},{}
    for e in edges:
        if e[0] == e[1]: continue
        if is_parent2child is True:
            e = e[::-1]
        if e[0] not in c2p:
            c2p[e[0]] = set([])
        if e[1] not in p2c:
            p2c[e[1]] = set([])
        c2p[e[0]].add(e[1])
        p2c[e[1]].add(e[0])

    # Then it recursively adds the parents of the parents (childs of the childs)
    def recursion(i, d, skip_terms = set([])):
        r = set([])
        try:
            members = set(d[i]) - (set([i]) | r)
        except KeyError:
            return r

        for j in members:
            if j in skip_terms: continue
            r.add(j)
            r.update(recursion(j, d)) #recursion

        return r

    result, mappings = [], [c2p, p2c]
    for mapping in mappings:
        d = {}
        for n in mapping:
            all_members = set([])
            for member in mapping[n]:
                all_members.add(member)
                all_members.update(recursion(member, mapping))
            if include_itself:
                all_members.add(n)
            d[n] = all_members
        result.append(d)

    for x in set(c2p)|set(p2c):
        for ix in range(len(result)):
            if x not in result[ix]:
                result[ix][x] = set([])
            if include_itself:
                result[ix][x].add(x)

    return tuple(result)

def get_the_youngest(nodes,c2p):
    return nodes.difference(set([i for x in nodes for i in c2p[x] if i != x])).pop()

def get_partners(n,mapping):
    partners = set([])
    for x in n:
        try:
            partners.update(mapping[x])
        except:
            continue
    return partners

def depropagate(m,h1=[],h2=[]):
    if len(h1) == len(h2) == 0:
        return {x:x for x in m}
    elif len(h1) == 0:
        return depropagate_n2(m,h2)
    elif len(h2) == 0:
        return depropagate_n1(m,h1)
    else:
        return depropagate_2_ontologies(m,h1,h2)

def propagate(m,h1=[],h2=[]):
    if len(h1) == len(h2) == 0:
        return m
    elif len(h1) == 0:
        return propagate_n2(m,h2)
    elif len(h2) == 0:
        return propagate_n1(m,h1)
    else:
        return propagate_2_ontologies(m,h1,h2)

def depropagate_n1(m,h):
    c2p,p2c = get_c2p_and_p2c(h)

    x2y = {}
    for x in m:
        if x[1] not in x2y:
            x2y[x[1]] = set([])
        x2y[x[1]].add(x[0])

    nonredundant_associations = set([])
    for x, ys in tqdm(x2y.items(),desc='Depropagating',leave=False):

        while len(ys) > 0:
            y = next(iter(ys))
            Y_offspring = p2c[y]
            Y_ancestors = c2p[y] #To remove ancestor associations

            Y_candidates = Y_offspring & ys

            if len(Y_candidates) == 1:
                nonredundant_associations.add((y,x))

            #Removing ancestors of the current y that are inse ys (since they will be always more redundant)
            ys = ys.difference(Y_ancestors)

    return nonredundant_associations

def depropagate_n2(m,h):
    c2p,p2c = get_c2p_and_p2c(h)

    x2y = {}
    for x in m:
        if x[0] not in x2y:
            x2y[x[0]] = set([])
        x2y[x[0]].add(x[1])

    nonredundant_associations = set([])
    for x, ys in tqdm(x2y.items(),desc='Depropagating',leave=False):

        while len(ys) > 0:
            y = next(iter(ys))
            Y_offspring = p2c[y]
            Y_ancestors = c2p[y] #To remove ancestor associations

            Y_candidates = Y_offspring & ys

            if len(Y_candidates) == 1:
                nonredundant_associations.add((x,y))

            #Removing ancestors of the current y that are inse ys (since they will be always more redundant)
            ys = ys.difference(Y_ancestors)

    return nonredundant_associations

def depropagate_2_ontologies(m,h1,h2):

    #Getting hierarchy connected matrix
    x_c2p,x_p2c = get_c2p_and_p2c(h1)
    y_c2p,y_p2c = get_c2p_and_p2c(h2)

    #Reading associations
    x2y = {}
    y2x = {}
    for x in m:
        if x[0] not in x2y:
            x2y[x[0]] = set([])
        x2y[x[0]].add(x[1])

        if x[1] not in y2x:
            y2x[x[1]] = set([])
        y2x[x[1]].add(x[0])

    #Iterating through the associations
    nonredundant_associations = set([])
    for x, ys in tqdm(x2y.items(),desc='Depropagating',leave=False): #Iterating using the x2y to better performance (define one time X data for many Y)

        #Get X_offspring_partners
        X_offspring = x_p2c[x]
        X_offspring_partners = get_partners(X_offspring,x2y)

        for y in ys:

            #Get Y offspring partners
            Y_offspring = y_p2c[y]
            Y_offspring_partners = get_partners(Y_offspring,y2x)

            #Checking if there are better Y_candidates (deeper in the ontology) than the current edge
            Y_candidates = X_offspring_partners & Y_offspring
            if len(Y_candidates) > 1: continue

            #Checking if there are better X_candidates (deeper in the ontology) than the current edge
            X_candidates = Y_offspring_partners & X_offspring
            if len(X_candidates) > 1: continue

            #If the script gets to here this means that the current association is not redundant
            nonredundant_associations.add((x,y))

    return nonredundant_associations

def propagate_n1(m,h):

    all_associations = set([])
    c2p,p2c = get_c2p_and_p2c(h)

    for n1,n2 in tqdm(m,desc='Propagating',leave=False):

        ancestors = c2p[n1]
        for ancs in ancestors:
             all_associations.add((ancs,n2))

    return all_associations

def propagate_n2(m,h):

    all_associations = set([])
    c2p,p2c = get_c2p_and_p2c(h)

    for n1,n2 in tqdm(m,desc='Propagating',leave=False):

        ancestors = c2p[n2]
        for ancs in ancestors:
            all_associations.add((n1,ancs))
    return all_associations

def propagate_2_ontologies(m,h1,h2):

    all_associations = set([])
    c2p_h1,p2c = get_c2p_and_p2c(h1)
    c2p_h2,p2c = get_c2p_and_p2c(h2)

    for n1,n2 in tqdm(m,desc='Propagating',leave=False):
        ancestors_n1 = c2p_h1[n1]
        ancestors_n2 = c2p_h2[n2]

        for ancs_n1 in ancestors_n1:
            for ancs_n2 in ancestors_n2:
                all_associations.add((ancs_n1,ancs_n2))

    return all_associations

def get_leaves(term,p2c):
    return set([t for t in p2c[term] if len(p2c[t]) == 1]).difference(set([term]))

def get_hyponyms(term,p2c):
    return set([t for t in p2c[term]]).difference(set([term]))

def get_subsumers(term,c2p):
    return set([x for x in c2p[term]]).difference(set([term]))

def get_maxterms(c2p):
    return set(c2p.keys())

def get_max_leaves(p2c):
    return set([t for t in p2c if len(p2c[t])==1])

def sanchez_term_IC(leaves,subsumers,max_leaves):
    """https://doi.org/10.1016/j.knosys.2010.10.001"""

    return -np.log(((leaves/subsumers)+1)/(max_leaves+1))

def seco_term_IC(hyponyms,maxterms):
    """http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.59.2199&rep=rep1&type=pdf"""

    return 1-(np.log(hyponyms+1)/np.log(maxterms))

def get_ontology_sanchez_IC(h):
    c2p,p2c = get_c2p_and_p2c(h)
    max_leaves = len(get_max_leaves(p2c))
    r = {}
    for term in p2c:
        leaves = len(get_leaves(term,p2c))
        subsumers = len(get_subsumers(term,c2p))
        r[term] = term_IC(leaves,subsumers,max_leaves)
    return r

def get_ontology_seco_IC(h):
    c2p,p2c = get_c2p_and_p2c(h)

    maxterms = len(get_maxterms(p2c))
    r = {}
    for term in p2c:
        hyponyms = len(get_hyponyms(term,p2c))
        r[term] = seco_term_IC(hyponyms,maxterms)
    return r
