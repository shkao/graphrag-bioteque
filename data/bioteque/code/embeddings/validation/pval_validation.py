root_path = '../../../'
code_path = root_path+'code/embeddings/'
emb_path = root_path+'embeddings/'

import sys
import os
import numpy as np
import pandas as pd
import h5py
import shutil
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(1, code_path)
import utils.graph_preprocessing as gpr
from validation.val_utils import *
from utils.utils import parse_parameter_file

## Variables
def get_parser():
    description = 'Runs metapath2vec'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-path', '--path',  type=str, required=True, help='Path of the embedding folder "current_path"')
    parser.add_argument('-d', '--distance', type=str, required=False, default='euclidean', help="Distance to be used (default=euclidean)")
    parser.add_argument('-pp', '--pp',  type=int, required=False, default=0, help='If 1 it uses matapath2vec++ embeddings (default=0)')
    parser.add_argument('-w', '--window', type=int, required=False, default=5, help='Window size used')
    parser.add_argument('-r', '--max_rk', type=float, required=False, default=0.75, help="Max p-value precision (default=0.75)")
    parser.add_argument('-e', '--pval_errors', type=str, required=False, default='0,0.001,0.0025,0.005,0.01,0.05,0.1', help="Errors to be considered (default=0,0.001,0.0025,0.005,0.01,0.05,0.1)")
    parser.add_argument('-wpcs', '--wpcs', type=str, required=False, default='0.05,0.1,0.25,0.5,0.75,1', help="Percentile groups used to group the neighbours if weighted (default=0.05,0.1,0.25,0.5,0.75,1)")
    parser.add_argument('-pneigh', '--pneigh', type=float, required=False, default=1, help='Proportion of real neighbours to be considered if weighted (default=1.0)')
    parser.add_argument('-mneigh', '--mneigh', type=int, required=False, default=5, help='Max number of neighbours to be considered if weighted (default=5)')

    return parser

args = get_parser().parse_args(sys.argv[1:])
current_path = args.path
distance = args.distance
pp = bool(args.pp)
window = args.window
pvalue_precision = args.max_rk
pval_errors = [float(x) for x in args.pval_errors.split(',')]
w_PCs = [float(x) for x in args.wpcs.split(',')]
prop_neigh_to_consider_if_weighted = args.pneigh
max_n_neigh_to_consider_if_weighted = args.mneigh
emb_path = '/'+'/'.join([x for x in current_path.split('/') if x!=''][:-3]) +'/'

quartiles = [0,25,50,75,100]
is_weighted = False #It will change to True if weights are detected in the edges

#--Validation path
validation_path = current_path +'/validation/'
if not os.path.exists(validation_path):
    os.mkdir(validation_path)
_wpp = 'w%i'%window if pp == False else '+w%i'%window
validation_path = validation_path+'/%s/'%_wpp
if not os.path.exists(validation_path):
    os.mkdir(validation_path)
validation_path = validation_path+'/distances/'
if not os.path.exists(validation_path):
    os.mkdir(validation_path)
validation_path = validation_path+'/%s/'%distance
if not os.path.exists(validation_path):
    os.mkdir(validation_path)

#--Checking metapath
mp,dt  = tuple(current_path.rstrip('/').split('/')[-2:])
with h5py.File(current_path+'/nd2st.h5','r') as f:
    nd2id = dict(zip(f['nd'][:].astype(str),f['id'][:].astype(str)))

_args = parse_parameter_file(current_path+'/parameters.txt')
mnd1 = _args['source']
mnd2 = _args['target']
if _args['tag_source']:
    mnd1 = _args['tag_source'] + mnd1
if _args['tag_target']:
    mnd2 = _args['tag_target'] + mnd2
mnds = [mnd1,mnd2]

#--Checking if is undirected
if mnd1 != mnd2 or (mnd1 == mnd2 and len(mp.split('-'))==3 and mp not in gpr._undirected_megs):
    is_undirected=False
else:
    is_undirected=True

#1) Getting labels, distances and ranks
if is_undirected:
    lbs1,ds1,rk1 = get_labels_dist_rank(mp, dt, mnd1=mnd1, mnd2=mnd2, distance=distance, max_nneigh=pvalue_precision, w= window, pp=pp, is_undirected=is_undirected, emb_path=emb_path)
    lbs1 = np.asarray([nd2id[n] for n in lbs1])
    lbs2 = lbs1
    rk2 = rk1
    uv1 = set(lbs1)
    del ds1

else:
    lbs1,ds1,rk1,lbs2,ds2,rk2 = get_labels_dist_rank(mp, dt, mnd1=mnd1, mnd2=mnd2, distance=distance, max_nneigh=pvalue_precision, w= window, pp=pp, is_undirected=is_undirected, emb_path=emb_path)
    lbs1 = np.asarray([nd2id[n] for n in lbs1])
    lbs2 = np.asarray([nd2id[n] for n in lbs2])
    del ds1, ds2
    uv1 = set(lbs1)
    uv2 = set(lbs1)

#2) Get neighbors and weights
n2neigh = gpr.read_node2neigh(current_path+'/edges.h5', map_nd=str)
if type(n2neigh) == tuple:
    n2neigh, n2w = n2neigh
    is_weighted = True

#3) Get p-values from neighs
n2pval = {n:[] for n in n2neigh}
n2pval_w = {n:[] for n in n2neigh}
for n in tqdm(list(n2neigh),desc='Getting p-values'):
    if n in uv1:
        lbs = [lbs1,lbs2]
        rk = rk1
    else:
        lbs = [lbs2,lbs1]
        rk = rk2

    source_ix = np.where(lbs[0]==n)[0][0]

    for i in range(len(n2neigh[n])):
        nn = n2neigh[n][i]
        nn_ix = np.where(lbs[1]==nn)[0][0]
        try:
            r = np.where(rk[source_ix] == nn_ix)[0][0]
            r = r/len(lbs[1])
        except IndexError:
            r = (rk.shape[1]+1)/len(lbs[1])

        n2pval[n].append(r)
        if is_weighted:
            n2pval_w[n].append(n2w[n][i])

del n2neigh

#5) Ploting

#--PLOT OPTIONS
#Fontname
fname ='Arial'
#FontSize
fsize = 16

#FONT
font = {'family' : fname,
        'size'   : fsize }

plt.rc('font', **font)

plt.rcParams['pdf.fonttype'] = 42
sns.set(font_scale=1.25,font="Arial")
sns.set_style("whitegrid")
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

#5.1) P-value error
M = [[]]
if not is_undirected:
    M.append([])

with open(validation_path+'/pval_errors.tsv','w') as o:
    o.write('mnd\tnode\t'+'\t'.join(['err_%s'%str(x) for x in pval_errors])+'\n')
    for i in range(len(M)):
        lb1 = lbs[i]
        lb2 = lbs[1-i]
        for n in lbs1:
            if is_weighted:
                pvs = np.asarray(n2pval[n])
                N = min([max_n_neigh_to_consider_if_weighted, round(len(pvs)*prop_neigh_to_consider_if_weighted)])
                W = np.asarray(n2w[n])
                cutoff = W[np.argsort(W)[::-1]][N-1]
                ix = np.where(W>=cutoff)
                pvs = pvs[ix]
            else:
                pvs = np.asarray(n2pval[n])

            cutoffs = [(len(pvs)+round((len(lb2)*err)))/len(lb2) for err in pval_errors]
            o.write('%i\t%s\t'%(i,str(n))+'\t'.join(['%.5f'%x for x in cutoffs])+'\n')
            M[i].append([sum(pvs<cutoff)/len(pvs) for cutoff in cutoffs])

with sns.axes_style("whitegrid"):
    if is_undirected:
        fig = plt.subplots(figsize=(7,5),dpi=100)
        ax1 = plt.subplot(111)
        axs = [ax1]
    else:
        fig = plt.subplots(figsize=(14,5),dpi=100)
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        axs = [ax1,ax2]
    yticks = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

    for i in range(len(M)):
        df = pd.DataFrame(np.asarray(M[i]), columns=[str(int(e*100))+'%' if int(e*100)!=0 else str(e*100)+'%' for e in pval_errors])

        sns.boxplot(data=df, ax=axs[i], flierprops={'marker':'o', 'markersize':2}, width=0.3, palette=col30)
        axs[i].set_xlabel('Error')
        axs[i].set_yticks(yticks)
        axs[i].set_title('%s (N=%i)'%(mnds[i],len(lbs[i])))
        axs[i].set_ylabel('Proportion of neighbors')

    if not is_undirected:
        ax2.set_ylabel('')
        ax2.set_yticklabels(['']*len(yticks))
        plt.subplots_adjust(wspace=0.01, hspace=0)

    plt.savefig(validation_path+'/error.png',bbox_inches='tight')

# 5.2) P-value dist
header_names = ['min' if x == 0 else 'max' if x==100 else 'Q%i'%x for x in quartiles]

# 5.2.1. Calculating p-value quartiles
if is_weighted:
    M = [[[] for _ in range(len(w_PCs))]]
    if not is_undirected:
        M.append([[] for _ in range(len(w_PCs))])
else:
    M = [[]]
    if not is_undirected:
        M.append([])

with open(validation_path+'/pval_quartiles.tsv','w') as o:

    if is_weighted:
        o.write('mnd\tnode\t'+'\t'.join(['W-%s_%s'%(str(x),y) for x in w_PCs for y in header_names]) +'\n')
    else:
        o.write('mnd\tnode\t'+'\t'.join(['%s'%x for x in header_names]))
    for i in range(len(M)): # n1 / n2
        for n in tqdm(lbs[i],desc='Getting p-values quartiles'): # Nodes
            if is_weighted:
                V = np.asarray(n2pval[n])
                W_ix_sort = np.argsort(n2pval_w[n])[::-1]
                rs = []
                for _ in range(len(w_PCs)):
                    prc = w_PCs[_]
                    v = V[W_ix_sort[:int(np.ceil(len(W_ix_sort)*prc))]]
                    r = [np.percentile(v,PC) for PC in quartiles]
                    M[i][_].append(r)
                    rs+=r
                o.write('%i\t%s\t'%(i,str(n))+'\t'.join(['%.5f'%x for x in rs])+'\n')

            else:
                r = [np.percentile(np.asarray(n2pval[n]),PC) for PC in quartiles]
                M[i].append(r)
                o.write('%i\t%s\t'%(i,str(n))+'\t'.join(['%.5f'%x for x in r])+'\n')

# 5.2.2. Plotting
with sns.axes_style("whitegrid"):
    if is_weighted:
        if is_undirected:
            fig = plt.subplots(figsize=(15,7),dpi=100)
            ax1 = plt.subplot(111)
            axs = [ax1]
        else:
            fig = plt.subplots(figsize=(15,14),dpi=100)
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212)
            axs = [ax1,ax2]
    else:
        if is_undirected:
            fig = plt.subplots(figsize=(6,5),dpi=100)
            ax1 = plt.subplot(111)
            axs = [ax1]
        else:
            fig = plt.subplots(figsize=(12,5),dpi=100)
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122)
            axs = [ax1,ax2]

    yticks = [0.01,0.05,0.1]+list(np.arange(0.2,pvalue_precision,0.1))
    if yticks[-1] != pvalue_precision:
        yticks.append(pvalue_precision)

    for i in range(len(M)):
        if is_weighted:
            m = []
            for i2,prc in enumerate(w_PCs):
                _m = np.asarray(M[i][i2])

                lb = '%i%%'%int(prc*100)

                for ix in range(_m.shape[1]):
                    m+=(zip(_m[:,ix],[header_names[ix]]*len(_m), [lb]*len(_m)))

            df = pd.DataFrame(m,columns=['P-value','Quartile','TOP prop. of neigh'])
            palette = dict(zip(header_names, col30))
            sns.boxplot(x='TOP prop. of neigh',y='P-value',hue='Quartile',data=df,flierprops={'marker':'o', 'markersize':2},ax=axs[i], palette=palette)

        else:
            m = np.asarray(M[i])
            df = pd.DataFrame(m,columns=header_names)
            sns.boxplot(data=df,flierprops={'marker':'o', 'markersize':2},ax=axs[i], palette=col30)

        axs[i].set_yticks(yticks)
        axs[i].set_title('%s'%(mnds[i]))
        axs[i].set_ylim(0,pvalue_precision+0.01)

    if not is_undirected and not is_weighted:

        axs[1].set_yticklabels(['']*len(yticks))
        plt.subplots_adjust(wspace=0.01, hspace=0)
        axs[0].set_ylabel('P-value')
        axs[0].set_xlabel('Neighbour Quartil')
        axs[1].set_xlabel('Neighbour Quartil')
plt.savefig(validation_path+'/pval_dist.png',bbox_inches='tight')
