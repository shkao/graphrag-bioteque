root_path = '../../../'
code_path = root_path+'code/embeddings/'
singularity_image = root_path+'/programs/ubuntu-py.img'

import os
import sys
import numpy as np
import h5py
import subprocess
import shutil
import argparse
sys.path.insert(0, code_path)
import skipgram.metapath4bioteque as mtp
from utils.utils import node2abbr

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
    parser.add_argument('-path', '--path',  type=str, required=True, help='Path of the embedding folder "current_path"')
    parser.add_argument('-pp', '--pp',  type=str, required=False, default='1', help='If 1 it runs matapath2vec++ (default=1)')
    parser.add_argument('-w', '--window', type=str, required=False, default='5', help='Window size used in the word2vec algorithm')
    parser.add_argument('-neg', '--negative', type=int, required=False, default=5, help="Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)")
    parser.add_argument('-s', '--size', type=int, required=False, default=128, help="Size of the embeddings (default=128")
    parser.add_argument('-thr', '--frequency_threshold', type=float, required=False, default=1e-3, help=" Set threshold for occurrence of words. Those that appear with higher frequency in the training data will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)")
    parser.add_argument('-threads', '--threads', type=int, required=False, default=16, help="Use <int> threads (default 16)")
    parser.add_argument('-it', '--iterations', type=int, required=False, default=5, help='Run more training iterations (default 5)')
    parser.add_argument('-min', '--min_count', type=int, required=False, default=5, help='This will discard words that appear less than <int> times; default is 5')
    parser.add_argument('-lr', '--learning_rate', type=float, required=False, default=0.025, help='Set the starting learning rate; default is 0.025 for skip-gram')
    parser.add_argument('-name', '--dataset_name',  type=str, required=False, default='m',  help='Name of the dataset inside h5df (default="m")')
    parser.add_argument('-run_in_cluster', '--run_in_cluster',  type=str2bool, nargs='?', const=True, default=True, help='Set it to True if you want to run in the cluster')

    return parser

args = get_parser().parse_args(sys.argv[1:])
#Variables
current_path = args.path
stats_path = current_path+'/stats'
walks_path = current_path+'/scratch/walks/_walks'
embedding_path = current_path+'/embs'
if not os.path.isdir(embedding_path):
    os.mkdir(embedding_path)

_scratch_path = current_path+'/scratch'
if not os.path.isdir(_scratch_path):
    os.mkdir(_scratch_path)
scratch_path = _scratch_path +'/embs/'
if not os.path.isdir(scratch_path):
    os.mkdir(scratch_path)

#Parameters
#--------------------------------------------------------------------------------------
#-- Default
n_walks = 1000
walk_length = 100
metapath = None
save_RAM_memory = False

#-- Updating parameters if provided
with open(current_path+'/parameters.txt','r') as f:
    for l in f:
        h = l.rstrip('\n').split('\t')
        #Number of walks
        if h[0] == 'nwalks':
            n_walks = int(h[1])
        #Walk length
        elif h[0] == 'length':
            walk_length = int(h[1])
        #Metapath
        elif h[0] == 'metapath':
            metapath = h[1:]

            if len(metapath) == 1:
                metapath = metapath[0]
                if '-' in metapath:
                    metapath = metapath.split('-')
                elif ',' in metapath:
                    metapath = metapath.split(',')
                else:
                    sys.exit('Unknown metapath separator: %s'%metapath)
            else:
                _ = set([])
                for mp in metapath:
                    _.add(len(mp.split('-')))
                assert len(_) == 1, 'The provided metapaths do not have the same length!\n'
                _ = [set([]) for i in range(_.pop())]

                for mp in metapath:
                    for ix, k in enumerate(mp.split('-')):
                        _[ix].add(k)

                metapath = []
                for x in _:
                    metapath.append('+'.join(sorted(x)))

        #Should we take care of the computer RAM memory?
        elif h[0] == 'save_ram_memory':
            if h[1].lower() == 'true' or h[1] == '1' or h[1].lower()== 't':
                save_RAM_memory = True
            elif h[1].lower() == 'false' or h[1] == '0' or h[1].lower()== 'f':
                save_RAM_memory = False
            else:
                sys.exit('Unknown save_ram_memory value: %s'%(h[1]))

#Node mapping
with h5py.File(current_path+'/nd2st.h5','r') as f:
    nd2id = dict(zip(f['nd'][:].astype(str), f['id'][:].astype(np.int)))

#--Node universe
nds = set([x[:-4] for x in os.listdir(current_path+'/nodes/')]) #Source and target nodes

nd_uv = {}
for nd in nds:
    nd_uv[nd] = set([])
    with open(current_path+'/nodes/%s.txt'%nd,'r') as f:
        for l in f:
            n  = l.rstrip()
            if n in nd2id:
                nd_uv[nd].add(nd2id[n])
#------------------------------------------------------------------------------

# 1) Preprocessing walks
if (not os.path.exists(current_path+'/walks.txt.gz')) and (not os.path.exists(current_path+'/walks.txt')):
    sys.stderr.write('Preprocessing rwalks...\n')
    sys.stderr.flush()
    error = mtp.preprocess_walks(walks_path, node_universe=nd_uv,
                                 write_to=current_path+'/walks.txt', compressed=True)
    if error != None:
        sys.exit('*ERROR*: %s. Exiting...\n'%error)
    else:
        #--Removing old walks
        shutil.rmtree(walks_path)
        #--Decompressing the walks (this will be removed at the end of the script)
        if not os.path.exists(current_path+'/walks.txt'):
            cmd = "zcat '%s/walks.txt.gz' > '%s/walks.txt'"%(current_path, current_path)
            subprocess.Popen(cmd, shell = True).wait()

# 2) Sanity Checking
if os.path.exists(current_path+'/stats/rwalk_quality.txt') is False:
    sys.stderr.write('Checking the rwalk paths...\n')
    sys.stderr.flush()
    mtp.walk_checker(current_path+'/walks.txt', nwalks=n_walks, length = walk_length, write_to=stats_path+'/rwalk_quality.txt')

    with open(stats_path+'/rwalk_quality.txt','r') as f:
        for l in f:
            h = l.rstrip().split(':')
            if h[0] == 'source_nodes':
                n_src = int(h[1].strip())
            if h[0] == 'corpus':
                corpus = int(h[1].strip())
        if n_src != corpus:
            sys.exit('*ERROR*: There was a problem in the walks as the full corpus (%i) is not represented as source nodes (%i). Exiting...\n'%(corpus, n_src))

# 3) Checking metapath2vec parameters
cpus = 16
outpath = scratch_path+'/emb'
cluster_params = {"num_jobs": 1,
                  "jobdir": scratch_path,
                  "job_name": "emb_mp2v",
                  "wait": True,
                  "memory": cpus*4,
                  "cpu": cpus,
                  "mem_by_core": 4
}

# 4) Running metapath2vec
sys.stderr.write('Running metapath2vec...\n')
sys.stderr.flush()

pps = [int(i.strip()) for i in args.pp.split(',')]
ws = [int(window.strip()) for window in args.window.split(',')]
names = [n.strip() for n in args.dataset_name.split(',')]

assert len(pps) == len(names)

for w in ws:
    for i in range(len(pps)):
        pp = pps[i]
        name = names[i]

        #--Running metapath2vec
        mtp.metapath2vec(current_path+'/walks.txt', outpath, pp=pp, size=args.size, window=w, sample=args.frequency_threshold, negative=args.negative,
        threads=args.threads, iterations=args.iterations, min_count = args.min_count, alpha=args.learning_rate,
        run_in_cluster = args.run_in_cluster,cluster_params=cluster_params)

        #--Making h5 file
        rows = []
        m = []

        #--Read universe for each nodes
        nd_emb = {nd:[] for nd in nds}
        nd_rows = {nd:[] for nd in nds}

        with open(outpath+'.txt','r') as f:
            f.readline()
            f.readline()
            for l in f:
                h = l.split(' ')
                lb = int(h[0][1:]) #Skipping the tag letter
                emb = np.asarray(h[1:-1]).astype(np.float)

                #classifying the nodes
                for nd in nd_uv:
                    if lb in nd_uv[nd]:
                        nd_rows[nd].append(lb)
                        nd_emb[nd].append(emb)
                        break

        for nd in nd_rows:
            r2m =  dict(zip(np.asarray(nd_rows[nd]), np.asarray(nd_emb[nd])))

            with open(current_path+'/nodes/%s.txt'%nd,'r') as f:
                m_sorted = np.asarray([r2m[nd2id[n]] for n in f.read().splitlines()]).astype(np.float32)

            #--Writing h5
            folder_name = 'w%i'%w
            #if pp == 1:
            #    folder_name = '+'+folder_name
            folder_name = embedding_path+'/%s'%folder_name
            if not os.path.isdir(folder_name):
                os.mkdir(folder_name)

            ofile = folder_name+'/%s.h5'%nd
#            if os.path.exists(ofile):
#                os.remove(ofile)
            with h5py.File(ofile,'a') as o:
                if name in o.keys():
                    del o[name]
                o.create_dataset(name, data=m_sorted)

        #--Removing metapath2vec out files
        os.remove(outpath)
        os.remove(outpath+'.txt')
