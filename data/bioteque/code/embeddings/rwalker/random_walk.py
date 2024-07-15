root_path = '../../../'
code_path = root_path+'code/embeddings/'

import sys
import os
from tqdm import tqdm
import argparse
import numpy as np
import h5py
sys.path.insert(1, code_path)
from utils.graph_preprocessing import read_edges

def get_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    description = 'Runs random walker'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-edges', '--edge_path', type=str, required=True, help='Path with the edges')
    parser.add_argument('-ofile', '--ofile', type=str, required=True, help='Path to write the edges')
    parser.add_argument('-weighted', '--is_weighted',  type=str2bool, nargs='?', const=True, default=False, help='If the graph is weighted')
    parser.add_argument('-direct', '--is_directed',  type=str2bool, nargs='?', const=True, default=False, help='If the graph is directed')
    parser.add_argument('-nwalks', '--nwalks',  type=int, required=False, default=100, help='(default 100) Number of walks')
    parser.add_argument('-length', '--length',  type=int, required=False, default=100, help='(default 100) Length of the walks')
    parser.add_argument('-nodes', '--nodes',  type=str, required=False, default=None, help='List of nodes to walk. Separated by comma. "n1,n2,n3..."')
    #parser.add_argument('-seed', '--seed',  type=int, required=False, default=42, help='(default 42) Numpy random seed')
    return parser

args = get_parser().parse_args()
#np.random.seed(args.seed)

#--Getting graph
sys.stderr.write('Reading the graph and preparing the walker...\n')
nd2e, nd2w = {}, {}

for h in read_edges(args.edge_path):

    n1 = int(h[0])
    n2 = int(h[1])
    if n1 not in nd2e:
        nd2e[n1] = []
    nd2e[n1].append(n2)

    if not args.is_directed:
        if n2 not in nd2e:
            nd2e[n2] = []
        nd2e[n2].append(n1)

    if args.is_weighted:
        if n1 not in nd2w:
            nd2w[n1] = []
        nd2w[n1].append(float(h[2]))

        if not args.is_directed:
            if n2 not in nd2w:
                nd2w[n2] = []
            nd2w[n2].append(float(h[2]))

#--Changing data types
nd2e = {n: np.asarray(nd2e[n], dtype=np.int) for n in list(nd2e)}

if args.is_weighted:
    nd2w = {n: np.asarray(nd2w[n],dtype=np.float32)/sum(nd2w[n]) for n in list(nd2w)}

#Initializing random walker
if args.nodes:
    nodes = list(map(int,args.nodes.split(',')))
else:
    nodes = sorted(nd2e)

if os.path.exists(args.ofile):
    os.remove(args.ofile)

sys.stderr.write('Starting the random walker...\n\t-is_weighted: %s\n\t-is_directed: %s\n\n'%(str(args.is_weighted),str(args.is_directed)))

#Running random walker
with open(args.ofile,'a') as o:

    #--weighted random walk
    if args.is_weighted:
        for n in tqdm(nodes):
            for _ in range(args.nwalks):
                current_node = n
                walks = [current_node]
                for __ in range(args.length-1):
                    current_node = np.random.choice(nd2e[current_node],p=nd2w[current_node])
                    walks.append(current_node)
                o.write(' '.join(map(str,walks))+'\n')

    #--unweighted random walk
    else:
        for n in tqdm(nodes):
            for _ in range(args.nwalks):
                current_node = n
                walks = [current_node]
                for __ in range(args.length):
                    current_node =  np.random.choice(nd2e[current_node])
                    walks.append(current_node)
                o.write(' '.join(map(str,walks))+'\n')
