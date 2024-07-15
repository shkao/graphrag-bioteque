root_path = '../../'
code_path = root_path+'code/embeddings/'
emb_path = root_path+'embeddings/'
graph_edges_path = root_path+'/graph/processed/propagated/'
singularity_image = root_path+'/programs/ubuntu-py.img'
hpc_path = root_path+'/programs/hpc/'

import sys
import os
import numpy as np
import h5py
import subprocess
import shutil
import argparse
sys.path.insert(0, hpc_path)
from hpc import HPC
from config import config as cluster_config
sys.path.insert(0, code_path)
from utils.utils import parse_parameter_file, mpath2medges, flip_medge

def mp2str(metapath):
    return '-'.join(metapath)

def str2bool(s):
    _s = str(s)
    if _s.lower() in set(['t','true']) or _s == '1':
        return True
    elif _s.lower() in set(['f','false', 'none']) or _s == '0':
        return False
    else:
        sys.exit('Unknown bool value: %s'%s)

def run_command(command):
    process = subprocess.Popen(command.split(' '), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        output = process.stdout.readline().decode('utf8')

        if process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc

def run_bioteque2embedding(final_path, force_rewrite=False, get_edges=True, run_rwalk=True, run_metapath2vec=True, run_validation=True, window='5',
                           emb_path = emb_path):

    # *********
    # * PATHS *
    # *********
    graph_path = graph_edges_path
    get_edges_script = code_path+'/build_network/get_edges.py'
    get_edges_from_precomputed_file_script = code_path+'/build_network/get_edges_from_precomputed_file.py'
    rwalk_script = code_path+'/rwalker/walks.py'
    mp2vec_script = code_path+'/skipgram/mp2vec.py'
    val_script = code_path+'/validation/run_validation.py'
    final_path = '/'+'/'.join([x for x in final_path.split('/') if x != ''])

    # *********************
    # Checking parameters *
    # *********************
    walk_length = 100
    nwalks = 100
    verbose = True
    pprob = 1
    qprob = 1
    metapath_name = None
    get_metapath_name = False
    save_RAM_memory = False
    precomputed_edges = False
    with open(final_path+'/parameters.txt','r') as f:
        for l in f:
            h = l.rstrip('\n').split('\t')

            #Metapath
            if h[0] == 'metapath':
                metapath = h[1:]
                if len(metapath) == 1:
                    metapath = metapath[0]
                else:
                    get_metapath_name = True

            #Input data
            elif h[0] == 'datasets':
                datasets = ['+'.join(x) for x in zip(*[x.split('||') for x in h[1:]])]

            elif h[0] == 'precomputed_edges':
                precomputed_edges = str2bool(h[1])

            #Number of walks
            elif h[0] == 'nwalks':
                nwalks = int(h[1])
            #Walk length
            elif h[0] == 'length':
                walk_length = int(h[1])

            #Metapath name
            elif h[0] == 'metapath_name':
                metapath_name = h[1].split('-')

            #Size of the node split (to paralellize in the cluster)
            elif h[0] == 'verbose':
                verbose = h[1]

            #n2v parameters
            elif h[0] == 'q':
                qprob = float(h[1])
            elif h[0] == 'p':
                pprob = float(h[1])

            #Should we take care of the computer RAM memory?
            elif h[0] == 'save_ram_memory':
                if h[1].lower() == 'true' or h[1] == '1' or h[1].lower()== 't':
                    save_RAM_memory = True
                elif h[1].lower() == 'false' or h[1] == '0' or h[1].lower()== 'f':
                    save_RAM_memory = False
                else:
                    sys.exit('Unknown save_ram_memory value: %s'%(h[1]))

    if metapath_name:
        metapath = metapath_name
    elif get_metapath_name:
        sys.exit('More than one metapath detected. You need to provide the final name of the metapath!')

    if type(metapath) == str:
        metapath = metapath.split('-')

    if type(window) == str:
        window = window.split(',')

    sys.stderr.write('Working on --> %s\n\n'%(' || '.join(final_path.rstrip('/').split('/')[-2:])))
    _source = metapath[0]
    _target = metapath[-1]

    # *******************************************************
    # Checking if files already exist. If True, remove them *
    # *******************************************************
    final_files = set([])

    if get_edges == True:
        final_files.update([final_path+'/edges.tsv', final_path+'/edges.h5', final_path+'/nd2st.h5', final_path+'/nodes', final_path+'/stats'])

    if run_rwalk == True:
        final_files.update([final_path+'/walks.txt',final_path+'/walks.txt.gz',final_path+'/walks/raw_walks.txt'])

    if run_metapath2vec == True:
        pp_sign =['']
        if _source != _target:
            pp_sign.append('+')
        final_files.update([final_path+'/embs/%sw%s'%(sign,w) for w in window for sign in pp_sign])

    if run_validation == True:
        final_files.update([final_path+'/validation',final_path+'/nneigh'])


    already_exist = set([ff for ff in final_files if os.path.exists(ff)])
    if len(already_exist) > 0:
        if force_rewrite == True:
            sys.stderr.write('Removing the following file/folders...\n\n- %s\n\n'%('\n- '.join(sorted(already_exist))))
            sys.stderr.flush()
            for ff in already_exist:
                if os.path.isdir(ff):
                    shutil.rmtree(ff)
                else:
                    os.remove(ff)
        else:
            sys.exit('The following files already exists. Set "force_rewrite" = True if you want to rewrite them:\n\n- %s\n\n'%('\n- '.join(sorted(already_exist))))

    # **********************
    # Creating directories *
    # **********************
    if not os.path.isdir(final_path+'/stats/'):
        os.mkdir(final_path+'/stats/')
    if not os.path.isdir(final_path+'/scratch'):
        os.mkdir(final_path+'/scratch')
    if not os.path.isdir(final_path+'/scratch/edges'):
        os.mkdir(final_path+'/scratch/edges')
    if not os.path.isdir(final_path+'/scratch/walks'):
        os.mkdir(final_path+'/scratch/walks')

    #*********************************
    # Printing and writing parameters
    # ********************************
    args = parse_parameter_file(final_path+'/parameters.txt', graph_edges_path=graph_edges_path)
    sys.stderr.write('The following parameters were found:\n\n')
    with open(final_path+'/scratch/_used_parameters.txt','w') as o:
        for k,v in args.items():
            sys.stderr.write("%s\t-->\t%s\n"%(k, str(v)))
            o.write("%s\t-->\t%s\n"%(k, str(v)))
        sys.stderr.write('\n')

    # ****************
    #  Getting edges *
    # ****************
    if get_edges == True:
        sys.stderr.write('Getting edge file...\n')
        sys.stderr.flush()

        if precomputed_edges:
            cmd = 'python3 %s %s'%(get_edges_from_precomputed_file_script, final_path)
            sc = run_command(cmd)
            if sc == 1:
                sys.exit('\n\nExiting due to an error from: >%s<\n\n'%get_edges_script)

        else:
            cluster = HPC(**cluster_config)
            #--Updating cluster params
            cluster_params = dict(
                job_name='matmult',
                jobdir=final_path+'/scratch/edges',
                cpu=4,
                memory = 16,
                mem_by_core = 4,
                wait=True)

            command = "singularity exec %s python3 %s %s"%(singularity_image, get_edges_script, final_path)
            cluster.submitMultiJob(command, **cluster_params)

    # ***************
    # Running rwalk *
    # ***************
    if run_rwalk == True:
        sys.stderr.write('Preparing the random walker..\n')
        sys.stderr.flush()

        if save_RAM_memory is False:
            rwalk_jobs = 4
        else:
            rwalk_jobs = 4

        #with open(final_path+'/edges.tsv','r') as f:
    #        if len(f.readline().split('\t')) > 2:
        with h5py.File(final_path+'/edges.h5','r') as f:
            if 'weights' in f.keys():
                is_weighted = 'True'
            else:
                is_weighted = 'False'

        cmd = 'python3 %s -i %s -o %s -wtype %s -n %i -l %i -w %s -v %s -njobs %i'%(rwalk_script,final_path+'/edges.h5', final_path, wtype, nwalks, walk_length,  is_weighted, verbose, rwalk_jobs)
        if wtype == 'n2v':
            cmd += ' -qprob %i -pprob %i'%(qprob,prob)
        sc = run_command(cmd)
        if sc == 1:
            sys.exit('\n\nExiting due to an error from: >%s<\n\n'%rwalk_script)

    # **********************
    # Running metapath2vec *
    # **********************
    if run_metapath2vec == True:
        sys.stderr.write('Getting embeddings from walk paths..\n')
        sys.stderr.flush()

        pp = ['0']
        names = ['m']
        cmd = 'python3 %s -path %s -pp %s -w %s -name %s'%(mp2vec_script,final_path,','.join(pp),','.join(window), ','.join(names))

        sc = run_command(cmd)
        if sc == 1:
            sys.exit('\n\nExiting due to an error from: >%s<\n\n'%mp2vec_script)

    #Compressing files
    #--Compressing walks
    if os.path.exists('%s/walks.txt'%final_path):
        if not os.path.exists('%s/walks.txt.gz'%final_path):
            subprocess.Popen('gzip %s/walks.txt'%final_path, shell = True).wait()
        else:
            os.remove('%s/walks.txt'%final_path)


    # ********************
    # Running validation *
    # ********************
    if run_validation == True:
        sys.stderr.write('Running validation scripts..\n')
        sys.stderr.flush()

        cmd = 'python3 %s %s'%(val_script,final_path)

        sc = run_command(cmd)
        if sc == 1:
            sys.exit('\n\nExiting due to an error from: >%s<\n\n'%val_script)

    # **************************************
    # Making symlink from target to source *
    # **************************************
    if _source != _target:
        reverse_path = '%s/%s/%s'%(emb_path,metapath[-1],'-'.join(metapath[::-1]))
        if not os.path.exists(emb_path+'/%s'%metapath[-1]):
            os.mkdir(emb_path+'/%s'%metapath[-1])
        if not os.path.islink(reverse_path):
            os.symlink(emb_path+'/%s/%s'%(metapath[0],'-'.join(metapath)),reverse_path)

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
    parser.add_argument('-path', '--final_path', type=str, required=True, help='Path with the parameters file to run random_walks and metapath2vec')
    parser.add_argument('-w', '--window', type=str, required=False, default='5', help='Window(s) size used in the word2vec algorithm. (default= "5")')
    parser.add_argument('-rewrite', '--force_rewrite',  type=str2bool, nargs='?', const=False, default=False, help='Set it to True if you want to rewrite the files')
    parser.add_argument('-get_edges', '--get_edges',  type=str2bool, nargs='?', const=True, default=True, help='Set it to True if you want to get edge file')
    parser.add_argument('-rwalk', '--run_rwalk',  type=str2bool, nargs='?', const=True, default=True, help='Set it to True if you want to run the random walker')
    parser.add_argument('-metapath2vec', '--run_metapath2vec',  type=str2bool, nargs='?', const=True, default=True, help='Set it to True if you want to run metapath2vec')
    parser.add_argument('-validation', '--run_validation',  type=str2bool, nargs='?', const=True, default=True, help='Set it to True if you want to run the validation script')

    return parser

if __name__ == "__main__":
    args = vars(get_parser().parse_args())
    run_bioteque2embedding(**args)
