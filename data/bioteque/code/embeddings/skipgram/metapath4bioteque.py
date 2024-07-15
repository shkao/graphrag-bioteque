root_path = '../../../'
code_path = root_path+'code/embeddings/'
scratch_path = root_path+'/scratch/'
singularity_image = root_path+'/programs/ubuntu-py.img'
hpc_path = root_path+'/programs/hpc/'
metapath2vec_path = root_path+'/programs/metapath2vec/'

import os
import sys
import numpy as np
import subprocess
import networkx as nx
from tqdm import tqdm
from collections.abc import Iterable
from collections import Counter

sys.path.insert(0, hpc_path)
sys.path.insert(0, code_path)
from utils.utils import mpath2medges, flip_medge

####################
# MAKING METAPATHS #
####################
def _required_node2vec_memory(gx):
    n2ns = {}
    for n in gx:
        n2ns[n] = len(gx[n])

    a = 0
    for n in gx:
        a+= (n2ns[n] + sum([n2ns[x] for x in gx[n]])) * 12

    return a/1e9


def MetaPathWalk(graph, metapath, length=100, nwalks=1000, nodes=None, type_attribute="label",
                 seed=None, weighted = False, write_to=None):
    """
    For heterogeneous graphs, it performs random walks based on a given metapath.

    Args:
            graph: <nx.classes.graph.Graph> nx graph with edge and node labels
            metapath: <list> List of node-edge-node labels that specify a metapath schema, e.g.,
            ['CPD','int','GEN','ppi','GEN','int','CPD']

            length: <int> Maximum length of each random walk
            nwalks: <int> Total number of random walks per root node
            nodes: <list> Subset of nodes to perform the random walk. If None it takes the whole graph
            type_attribute: <str> The node/edge attribute name that stores the node/edge's type
            seed: <int> Random number generator seed; default is None
            weighted: <bool> Use the edge weights as probabilities
            write_to: <str> Output file where write the paths. If None, the walks are returned as a list of lists.

        Returns:
            <list> List of lists of nodes ids for each of the random walks generated. If write_to is not None, it returns None.
    """
    #Variables
    walks = []

    # Preprocessing attributes values
    #-----------------------------------------------
    # 1) graph
    if type(graph) != nx.classes.multigraph.MultiGraph:
        raise ValueError('The graph must be a nx MultiGraph type.')

    # -- Mapping graph to integer to reduce memory
    nd2int = dict(zip(graph.nodes,np.arange(len(graph.nodes))))
    int2nd = {v:k for k,v in nd2int.items()}
    graph = nx.relabel_nodes(graph,nd2int)

    # 2) metapath
    if type(metapath) != list and type(metapath) != np.ndarray:
        raise ValueError('The metapaths parameter must be a list.')
    if metapath[0] != metapath[-1]:
        raise ValueError('The first and last node type in a metapath should be the same.')
    if len(metapath) < 3:
        raise ValueError("Each metapath must specify at least two node types and one edge, ex: ['GEN','ppi','GEN']")
    if len([x for x in metapath if type(x) != str]) > 0:
        raise ValueError('Node labels in metapaths must be string type.')

    # 3) length
    if type(length) != int or length < 0:
        raise ValueError('The walk length, length, should be positive integer.')

    # 4) nwalks
    if type(nwalks) != int or length < 0:
        raise ValueError('The number of walks per root node, n, should be a positive integer.')

    # 5) nodes
    if nodes is None:
        nodes = np.asarray(graph.nodes)

    else:
        if not isinstance(nodes, Iterable):
            raise ValueError('The nodes parameter should be an iterable of node IDs.')
        if len(nodes) == 0:
            raise ValueError('No starting node IDs given.')
        nodes = [nd2int[x] for x in nodes if x in nd2int] #If a provided node is not in the graph it is skipped

    # 6) seed
    if seed is not None:
        if type(seed) != int or seed < 0:
            raise ValueError('The random number generator seed value, seed, should be positive integer or None.')
    rs = np.random.RandomState(seed)

    # 7) write_to
    if write_to is not None:
        O = open(write_to,'w')
    #------------------------------------------------

    #Iterating across nodes
    for node in tqdm(nodes):

        # Retrieve node type
        label = graph.node[node][type_attribute]

        # If the label of the source node doesn't match with the metapath, skip it
        if label != metapath[0]:continue

        node_path = [x for ix,x in enumerate(metapath) if ix%2 == 0]
        node_path = node_path[1:] * ((length // (len(node_path) - 1)) + 1) #Skipping source node
        edge_path = [x for ix,x in enumerate(metapath) if ix%2 != 0]
        edge_path = edge_path * (len(node_path)//len(edge_path))

        # Iterating across the number of desired walks per node
        for _ in range(nwalks):
            walk = []  # holds the walk data for this walk; first node is the starting node
            current_node = node

            # Iterating across the metapath until the walk gets the desired length
            for d in range(length):
                walk.append(int2nd[current_node])

                # Getting neighbours connected by the desired edge and are from the desired type (according to the metapath)
                neighbours = [node
                              for node in graph[current_node]
                              if edge_path[d] in graph[current_node][node]
                              and graph.node[node][type_attribute] == node_path[d]]

                if len(neighbours) == 0:
                    # If no neighbours of the required type as dictated by the metapath exist, then stop.
                    break

                # Randomly picking up a neighbour. If the graph is weighted, it samples according to the weight
                if weighted is True:
                    probabilities = np.asarray([graph[current_node][neigh][edge_path[d]]['weight']
                                                for neigh in neighbours])
                    probabilities = probabilities/sum(probabilities) # assures that probabilities are in range 0-1
                else:
                    probabilities = None

                current_node = rs.choice(neighbours, p=probabilities)  # the next node in the walk

            # Stores the node in a file (if write_to is specified) or in memory (if write_to is None)
            if write_to is None:
                walks.append(walk)
            else:
                O.write('\t'.join(map(str,walk))+'\n')

    # Returning the walks if they were kept in memory. If not it closes the file and return None.
    if write_to is None:
        return walks
    else:
        O.close()
        return None

def walk_checker(walk_path,nwalks,length,verbose=True,write_to=None, delimiter=None):
    """
    Given file with random walks and the expected number of walks and length,
    it draws som statistics about the number of interrumped walks.
    If write_to is given it writes the statistics to the given path.
    """
    ranges = [int(np.percentile(range(length+1),x)) for x in [25,50,75,100]]
    stats = {r:[] for r in ranges}

    if write_to is not None:
        O = open(write_to,'w')

    #Calculate percentiles:
    np.percentile(range(length),25)
    ws = []
    #If the given path is a dir, read and merge all the files inside it
    if os.path.isdir(walk_path):
        files = [walk_path+f for f in os.listdir(walk_path) if os.path.isfile(walk_path+'/%s'%f)]
    #If the given path is a file, read it
    else:
        files = [walk_path]

    #Iterating
    nodes = set([])
    walks = 0
    words = 0
    corpus = set([])
    for file in files:
        if not os.path.isfile(file): continue
        with open(file,'r') as f:
            for l in f:
                if delimiter == None:
                    ws = l.rstrip('\n').split()
                else:
                    ws = l.rstrip('\n').split(delimiter)

                n = len(ws)
                walks +=1
                words += n
                nodes.add(ws[0])
                corpus.update(ws)

                if n == length:
                    continue
                else:
                    for cutoff in ranges:
                        if n <= cutoff:
                            stats[cutoff].append(ws[0])
                            break

    #Reporting
    if verbose is True:
        sys.stderr.write('Interrumped walks report:\n')
        sys.stderr.write('source_nodes: %i\n'%len(nodes))
        sys.stderr.write('walk_per_source: %i\n'%nwalks)
        sys.stderr.write('walk_length: %i\n'%length)
        sys.stderr.write('corpus: %i\n'%len(corpus))
        sys.stderr.write('walks: %i\n'%walks)
        sys.stderr.write('words: %i\n\n'%words)


    if write_to is not None:
        O.write('source_nodes: %i\n'%len(nodes))
        O.write('walk_per_source: %i\n'%nwalks)
        O.write('walk_length: %i\n'%length)
        O.write('corpus: %i\n'%len(corpus))
        O.write('walks: %i\n'%walks)
        O.write('words: %i\n\n'%words)

    c = 0
    q = ['1Q','2Q','3Q','4Q']
    m,m2 = [],[]
    for ix,cutoff in enumerate(sorted(stats)):
        Q = q[ix]
        c+= len(stats[cutoff])

        if verbose is True:
            sys.stderr.write('%s (<%s): %i walks from %i different nodes\n'%(Q,ranges[ix],len(stats[cutoff]),len(set(stats[cutoff]))))
        if write_to is not None:
            m.append([Q,ranges[ix],len(stats[cutoff]),len(set(stats[cutoff])),', '.join(['%s (%i)'%(r[0],r[1]) for r in Counter(stats[cutoff]).most_common()])])

    #Total interrumped
    if verbose is True:
        sys.stderr.write('Total interrumped walks: %i\n\n'%c)
    if write_to is not None:
        O.write('Total interrumped walks: %i\n\n'%c)
        for r in m:
            O.write('%s (<%s): %i walks from %i different nodes\n'%(r[0],r[1],r[2],r[3]))
            O.write('%s\n\n'%r[4])

        #Ranking
        O.write('Node Ranking:\n')

        total = []
        for r in stats.values():
            total += r
        O.write('\n'.join(['%s (%i)'%(r[0],r[1]) for r in Counter(total).most_common()]))

        O.close()

def preprocess_walks(walks, node_universe, node_mapping= None, mnd_to_remove = None,
                     write_to=None, compressed=True, delimiter=None):

    """
    This script preprocess the output of the randomwalk to be used for metapath2vec++
    It basically adds a prefix ("v","a","i","f") to differentiate the different metanodes in the walks

    Args:
        walks: file/dir path where the walks are.
        node_universe: dictionary mapping metanodes to the node id universe.
        node_mapping: dictionary mapping oficial Bioteque nodes into the labels used in the walks.
        mnd_to_remove: If any it remove those metanodes in the preprocessed walks.
        write_to: output file where the preprocess_walks should be written. If None it just returns a list of list with the preprocessed walks.
        compressed: if True the file is writed as gz
        delimiter: delimiter used in the walks file. If None it assumes to be regular spaces.
    """

    #Getting nd2lt
    lts = ['v','a','i','f']
    mnd2lt = dict(zip(list(node_universe), lts))
    nd2lt = {}
    for mnd in set(node_universe):
        if mnd_to_remove and mnd in mnd_to_remove:
            sys.stderr.write('Metanode %s was removed.\n'%mnd)
            continue

        nodes = node_universe[mnd]
        if node_mapping != None:
            nodes = [node_mapping[n] for n in nodes if n in node_mapping]
        nodes = list(map(str,list(nodes)))

        nd2lt.update(dict(zip(nodes,[mnd2lt[mnd]]*len(nodes))))

    if write_to is not None:
        if compressed:
            import gzip
            write_to = write_to+'.gz'
            O = gzip.GzipFile(write_to,'wb')
        else:
            O = open(write_to,'w')

    #Proprocessing
    wlkss = [] # list of walks

    #--walks is a path
    if type(walks) == str:
        #If the given path is a dir, read and merge all the files
        if os.path.isdir(walks):
            files = sorted([walks+'/%s'%f for f in os.listdir(walks) if os.path.isfile(walks+'/%s'%f)])
            if len(files) == 0:
                O.close()
                os.remove(write_to)
                return 'Not walks found in directory "%s".'%walks

            for file in tqdm(files, desc='Reading walk files...', leave=False):
                with open(file,'r') as f:
                     for l in tqdm(f, desc='Preprocessing', leave=False):
                        if delimiter == None:
                            ws = l.rstrip('\n').split()
                        else:
                            ws = l.rstrip('\n').split(delimiter)

                        ws = [nd2lt[w]+w for w in ws]
                        if write_to is not None:
                            if compressed:
                                O.write(b'\t'.join(map(str.encode,ws))+b'\n')
                            else:
                                O.write('\t'.join(ws)+'\n')
                        else:
                            wlkss.append(ws)

        #If the given path is a file, read it
        elif os.path.isfile(walks):
            with open(walks,'r') as f:
                c = 0
                for l in tqdm(f, desc='Preprocessing', leave=False):
                    c+=1
                    if delimiter == None:
                        ws = l.rstrip('\n').split()
                    else:
                        ws = l.rstrip('\n').split(delimiter)

                    ws = [nd2lt[w]+w for w in ws]
                    if write_to is not None:
                        if compressed:
                            O.write(b'\t'.join(map(str.encode,ws))+b'\n')
                        else:
                            O.write('\t'.join(ws)+'\n')
                    else:
                        wlkss.append(ws)
            if c == 0:
                O.close()
                os.remove(write_to)
                return 'The file "%s" does not contain any walk.'%walks

    #--walks is list of walks
    else:
        if len(walks) == 0:
            O.close()
            os.remove(write_to)
            return 'The given walk list is empty.'

        for ws in walks:
            ws = [nd2lt[w]+w for w in ws]
            if write_to is not None:
                if compressed:
                    O.write(b'\t'.join(map(str.encode,ws))+b'\n')
                else:
                    O.write('\t'.join(w_r)+'\n')
            else:
                wlkss.append(ws)


    if write_to is None:
        return wlkss
    else:
        O.close()

#####################
# MAKING EMBEDDINGS #
#####################
def metapath2vec(walks_path,output,pp=1,size=128,window=5,sample=0.001,negative=5,threads=32,iterations=5,min_count=5,
                 alpha=0.025,classes=0,debug=2,save_vocab=None,read_vocab=None,
                 run_in_cluster=False,cluster_config=None,cluster_params=None):
    """
    Options for training:
        -walks_path <file>
            Use text data from <file> to train the model
        -output <file>
            Use <file> to save the resulting word vectors / word clusters
        -pp <int>
            Use metapath2vec++ or metapath2vec; default is 1 (metapath2vec++); for metapath2vec, use 0
        -size <int>
            Set size of word vectors; default is 128
        -window <int>
            Set max skip length between words; default is 5
        -sample <float>
            Set threshold for occurrence of words. Those that appear with higher frequency in the training data
            will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)
        -negative <int>
            Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)
        -threads <int>
            Use <int> threads (default 12)
        -iter <int>
            Run more training iterations (default 5)
        -min_count <int>
            This will discard words that appear less than <int> times; default is 5
        -alpha <float>
            Set the starting learning rate; default is 0.025 for skip-gram
        -classes <int>
            Output word classes rather than word vectors; default number of classes is 0 (vectors are written)
        -debug <int>
            Set the debug mode (default = 2 = more info during training)
        -save_vocab <file>
            The vocabulary will be saved to <file>
        -read_vocab <file>
            The vocabulary will be read from <file>, not constructed from the training data

    Examples:
    ./metapath2vec -train ../in_dbis/dbis.cac.w1000.l100.txt -output ../out_dbis/dbis.cac.w1000.l100 -pp 1 -size 128 -window 7 -negative 5 -threads 32

    Input:
        The paths generated by random walks, each of which consists of different types of nodes. An example can be found in ../in_dbis/dbis.cac.w1000.l100.txt
        Currently, the code supports four types of nodes:
            1. one starting from "v", e.g., vKDD
            2. one starting from "a", e.g., aJiaweiHan
            3. one starting from "i"
            4. one starting from "f"

    Output:
        1. one vector file for each node in binary format (e.g., dbis.cac.w1000.l100)
        2. the same vector file for each node in text format (e.g., dbis.cac.w1000.l100.txt)
    """

    #Printing parameters
    plusplus = ''
    if pp == 1:
        plusplus = '++'

    sys.stderr.write(
    """
    ******************************
    *** Running metapath2vec%s ***
    ******************************

    >Parameters
    -------------------------
    -train: %s
    -output: %s
    -size: %i
    -window: %s
    -freq. threshold: %f
    -negative: %i
    -threads: %i
    -iter: %i
    -min-count: %i
    -lr: %f
    -classes: %i
    \n"""%(plusplus, walks_path, output, size, window, sample, negative, threads, iterations, min_count, alpha, classes))
    sys.stderr.flush()
    #Making the command query
    cmd = "%s/metapath2vec -train %s -output %s -pp %i -size %i -window %s -sample %f -negative %i -threads %i -iter %i -min-count %i -alpha %f -classes %i -debug %i"%(metapath2vec_path,walks_path,output,pp,size,window,sample,negative,threads,iterations,min_count,alpha,classes,debug)

    if save_vocab is not None:
        cmd += " -save-vocab %s"%save_vocab
    if read_vocab is not None:
        cmd += " -read-vocab %s"%read_vocab

    #Executing metapath2vec
    if run_in_cluster:

        from hpc import HPC
        if cluster_config is None:
            from config import config as cluster_config
        cluster = HPC(**cluster_config)

        if cluster_params is None:
            cluster_params = {}
            cluster_params["num_jobs"] = 1
            cluster_params["jobdir"] = scratch_path
            cluster_params["job_name"] = "emb_mpath2vec"
            cluster_params["wait"] = True
            cluster_params["memory"] = 64
            cluster_params["cpu"] = 32

        command = "singularity exec {} {}".format(singularity_image, cmd)
        cluster.submitMultiJob(command, **cluster_params)


    else:
        subprocess.Popen(cmd, shell = True).wait()
