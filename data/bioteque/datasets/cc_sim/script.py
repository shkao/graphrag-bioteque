import sys
import os
import numpy as np
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

#                      --------------------------------------------------------------
#---------------------- To run this you need access to the Chemical Checker database --------------------------------------------------------
#                      --------------------------------------------------------------
# """
# The 100 nearest neighbours for each CPD  in the universe were previously computed using the Chemical Checker stacked signature (25 spaces x 128 dimensions)
# Results were saved in './100NN.tsv'
# """
#
# #-----------------Parameters-------------------
#
# k = 10 #number of nearest neighboors
# max_dist = 0.5 # ~ Corresponds to approximately the TOP closest 1% of the background distance distribution 
#
# #----------------------------------------------
#
# current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
# source = current_path.split('/')[-1]
#
# with open('../../../graph/nodes/Compound.tsv','r') as f:
#     next(f)
#     u = sorted(set([l.rstrip().split('\t')[0] for l in f]))
#
# #--Getting edges from metanode similarities
# edges = set([])
# with open('./100NN.tsv', 'r') as f:
#     for l in f:
#         h = l.rstrip().split('\t')
#         i = h[0]
#         n = np.asarray(h[1].split('|'))
#         d = np.asarray(h[2].split('|'), dtype=float)
#
#         ixs = d<=d[k]
#         close_neigh = d<max_dist
#         ixs = np.where(np.logical_and(ixs,close_neigh))[0]
#         edges.update([tuple(sorted([i, n[ix]])+[d[ix]]) for ix in ixs])
#
# #Writing
# with open('./edges.tsv','w') as o:
#     o.write('n1\tn2\tcosine_dist\n')
#     for e in sorted(edges):
#         if e[0] == e[1]:continue
#         if e[0] not in u or e[1] not in u: continue #Keeping edges only in the universe
#         o.write('%s\t%s\t%.4f\n'%(e[0],e[1],e[2]))

#--------------------------------------------------------------------------------

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]
opath = out_path+'/CPD-sim-CPD/%s.tsv'%source


if os.path.exists(opath):
    if os.path.islink(opath):
        os.unlink(opath)
    else:
        os.remove(opath)

os.symlink(os.path.abspath('./edges.tsv'),os.path.abspath(opath))
