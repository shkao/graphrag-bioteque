import gzip
import shutil
import sys
import os
import numpy as np

sys.path.insert(0, "../../code/kgraph/utils/")
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

current_path = os.path.dirname(os.path.realpath(__file__))
source = os.path.basename(current_path)
output_directory = "../../graph/raw/CPD-sim-CPD"
os.makedirs(output_directory, exist_ok=True)
output_file_path = os.path.join(output_directory, f"{source}.tsv")

with gzip.open("./edges.tsv.gz", "rb") as f_in, open(output_file_path, "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)
    print(f"File copied successfully to {output_file_path}")

with open(output_file_path, "r") as f:
    next(f)  # Skip the header
    edges_count = sum(1 for _ in f)
    print(f"{edges_count} edges have been parsed and saved.")
