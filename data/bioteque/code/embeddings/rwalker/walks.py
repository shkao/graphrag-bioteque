root_path = '../../../'
code_path = root_path+'code/embeddings/'
scratch_path = root_path+'/scratch/'
singularity_image = root_path+'/programs/ubuntu-py.img'
hpc_path = root_path+'/programs/hpc/'

import sys
import os
import argparse
import subprocess
import numpy as np
import h5py
import shutil
sys.path.insert(0, hpc_path)
from hpc import HPC
from config import config as cluster_config
sys.path.insert(1, code_path)
from utils import graph_preprocessing as gpr
from utils.utils import *

def write_run_rwalk_script(path):
    tmp_script = """
import sys
import pickle
import subprocess

task_id = sys.argv[1]  # <TASK_ID>
filename = sys.argv[2]  # <FILE>
inputs = pickle.load(open(filename, 'rb'))  # load
rwalk_args = inputs[task_id][0]

args = '-edges %s -ofile %s -nwalks %s -length %s'%(rwalk_args['edges'], rwalk_args['ofile'], str(rwalk_args['nwalks']), str(rwalk_args['length']))

if rwalk_args['is_weighted']:
    args+= ' --is_weighted'
if rwalk_args['is_directed']:
    args+= ' --is_directed'
args+= ' -nodes %s'%(','.join(map(str,rwalk_args['nodes'])))

cmd = 'python3 {} %s'%args
subprocess.Popen(cmd, shell = True).wait()
""".format(code_path+'/rwalker/random_walk.py')
    with open(path,'w') as o:
        o.write(tmp_script)

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

    description = 'Runs metapath2vec'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--edge_path',  type=str, required=True, help='Path of edges. Can be a edge file (.tsv, .csv) or adjancy matrix (.h5)')
    parser.add_argument('-o', '--out_path',  type=str, required=True, help='Path to write the output walks')
    parser.add_argument('-n', '--nwalks', type=int, required=False, default=100, help="Number of walks per node (default: 100)")
    parser.add_argument('-l', '--walk_length', type=int, required=False, default=100, help="Length of the walk (default: 100)")
    parser.add_argument('-w', '--is_weighted',  type=str2bool, nargs='?', const=True, default=True, help='Set it to True to consider weights from the edge file (default:True)')
    parser.add_argument('-v', '--verbose',  type=str2bool, nargs='?', const=True, default=True, help='Verbose flag (default is True)')
    parser.add_argument('-qprob', '--qprob', type=float, required=False, default=1, help="q_prob used in n2v (default:1)")
    parser.add_argument('-pprob', '--pprob', type=float, required=False, default=1, help="p_prob used in n2v (default:1)")
    parser.add_argument('-njobs', '--njobs', type=int, required=False, default=4, help="Number of jobs (default=4")

    return parser
args = get_parser().parse_args(sys.argv[1:])

#Reading parameters
edge_list_path = args.edge_path
out_path = args.out_path
n_walks = args.nwalks #default 100
walk_length = args.walk_length #default 100
is_weighted = args.is_weighted #default True
verbose = args.verbose #default True
p_prob = args.pprob #default 1
q_prob = args.qprob #default 1
n2v_mem_cutoff = 30
njobs = args.njobs

#1) Reading and parsing input variables
sys.stderr.write('Reading and parsing input variables...\n')

#--1.1. Bulding directory system
_scratch_path = out_path+'/scratch'
if not os.path.isdir(_scratch_path):
    os.mkdir(_scratch_path)
scratch_path=  _scratch_path+'/walks/'
if not os.path.isdir(scratch_path):
    os.mkdir(scratch_path)
walks_path = scratch_path + '/_walks/'
if not os.path.exists(walks_path):
    os.mkdir(walks_path)
output_walks_path = walks_path+'/raw_walks.txt'


#Checking which random walker to run

#--Random walker

sys.stderr.write('Running random walker...\n')
sys.stderr.flush()
#Getting parameters for the rwalk (copyin the same)
rwalk_parm = dict(
    edges= edge_list_path,
    is_weighted= is_weighted,
    length= walk_length,
    nwalks= n_walks,
    is_directed = False)

#--Making elements
if edge_list_path.endswith('.tsv'):
    _nodes = set([])
    with open(edge_list_path, 'r') as f:
        for l in f:
            _nodes.update(l.rstrip().split('\t')[:2])
elif edge_list_path.endswith('.h5'):
    with h5py.File(edge_list_path, 'r') as f:
        _nodes = np.unique(f['edges'][:].ravel())
_nodes = sorted(_nodes)

_nodes_list =  []
elements = []
nodes_per_walk = 1000
for i in np.arange(0,len(_nodes)+1,nodes_per_walk):
    _d = {}
    _d.update(rwalk_parm)
    _d.update({'nodes': _nodes[i:i+nodes_per_walk]})
    _d.update({'ofile': walks_path+'/%i_%i.txt'%(i+1,min(i+nodes_per_walk+1,len(_nodes)))})
    elements.append(_d)

#--Writing temporary script
write_run_rwalk_script(scratch_path+'/tmp_script.py')

#Getting cluster parameters

cluster = HPC(**cluster_config)
#--Updating cluster params
cluster_params = {}
cluster_params['job_name'] = 'rwalk'
cluster_params["jobdir"] = scratch_path
#cluster_params["memory"] =  njobs
cluster_params['cpu'] = njobs
#cluster_params['mem_by_core'] = 4
cluster_params["wait"] = True
cluster_params["elements"] = elements
cluster_params["num_jobs"] = len(elements)

command = "singularity exec {} python {} <TASK_ID> <FILE>".format(
singularity_image,
scratch_path+'/tmp_script.py')
cluster.submitMultiJob(command, **cluster_params)

os.remove(scratch_path+'/tmp_script.py')
