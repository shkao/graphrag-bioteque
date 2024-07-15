root_path = '../../../../'
code_path = root_path+'code/embeddings/'
emb_path = root_path+'embeddings/'

import sys
import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean, cosine
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
sys.path.insert(1, code_path)
from utils.utils import get_undirected_megs, parse_parameter_file
from utils.embedding_utils import read_embedding, get_embedding_universe
from validation.val_utils import col30
from utils.graph_preprocessing import get_neg_edges_from_egs_simple, read_edges

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

#-----------------

def get_emb(mpath, dt, mnds, emb_path):
    emb = {}
    for mnd in set(mnds):
        emb.update(read_embedding(mpath=mpath, dt=dt, mnd=mnd, w=5, pp=False, emb_path=emb_path))
    return emb

def get_val_edges(path, universe=None):

    edge_path = path+'/edges.h5'
    id2nd_path = path+'/nd2st.h5'
    with h5py.File(id2nd_path, 'r') as f:
        id2nd = dict(zip(f['id'][:].astype(int), f['nd'][:].astype(str)))

    is_w = False
    nd2ew = {}
    edges = set([])

    n_edges_val = 0
    covered_edges = 0
    cov_uv = set([])

    for h in read_edges(edge_path):
        n_edges_val+=1
        n1 = int(h[0])
        n2 = int(h[1])

        if universe and (id2nd[n1] not in universe or id2nd[n2] not in universe): continue
        covered_edges+=1
        cov_uv.update([n1,n2])

        if len(h) >2:
            is_w = True
        if is_w:
            if n1 not in nd2ew:
                nd2ew[n1] = []
            if n2 not in nd2ew:
                nd2ew[n2] = []

            nd2ew[n1].append((n2,float(h[2])))
            nd2ew[n2].append((n1,float(h[2])))
        else:
            edges.add((n1,n2))

    #Getting coverages
    n_nodes_val = len(id2nd)
    cov_nodes_val = len(cov_uv) / n_nodes_val
    cov_uv = len(cov_uv)/len(universe)
    cov_edges_val = covered_edges/n_edges_val

    if is_w:
        #--Sorting
        nd2ew = {n: np.asarray(sorted(nd2ew[n], key = lambda x: x[1], reverse=True)) for n in nd2ew}

        for name,prc in [('w-max',0), ('w-25%',0.25), ('w-50%',0.5), ('w-100%',1)]:
            edges = set([])
            for n in nd2ew:
                cutoff =nd2ew[n][max([0,int(len(nd2ew[n])*prc)-1]),1]
                gns = [id2nd[x] for x in nd2ew[n][np.where(nd2ew[n][:,1] >= cutoff)[0],0]]
                edges.update(zip([id2nd[n]]*len(gns), gns))

            yield  name, edges, cov_uv, cov_nodes_val, n_nodes_val, cov_edges_val, n_edges_val
    else:
        yield  '', set([(id2nd[e[0]], id2nd[e[1]]) for e in edges]) , cov_uv,  cov_nodes_val, n_nodes_val, cov_edges_val, n_edges_val

def plot_roc(path, opath, figsize=(4.2,4), dpi=100, title='', nd_cov=None):

    aurocs = {}

    #Ploting roc
    weight_ratios = [9.5, 0.5]
    gridspec_kw = {"width_ratios" : weight_ratios}
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=figsize, gridspec_kw=gridspec_kw)
    plt.subplots_adjust(wspace=0.0, hspace=0)

    files = np.unique([x.split('_roc_')[-1].split('.')[0] for x in os.listdir(path) if x.startswith('_roc_')])

    for ix, n in enumerate(files):
        _aurocs = []
        for dist, ls in [('cos', ':'), ('euc','-')]:
            file = path+'_roc_'+n+'.%s'%dist

            with open(file,'r') as f:
                fpr = [float(x) for x in f.readline().rstrip().split('\t')]
                tpr = [float(x) for x in f.readline().rstrip().split('\t')]
                _aurocs.append('%.2f'%auc(fpr,tpr))

                if dist == 'euc':
                    label = '%s|%s%s'%(_aurocs[0], _aurocs[1], ' (%s)'%n if n!='' else '')
                else:
                    label = None
                axes[0].plot(fpr, tpr, color=col30[ix], linestyle=ls, lw=2, label = label)

            os.remove(file)

        aurocs[n] = '|'.join(_aurocs)

    #Ploting coverages
    axes[1].bar([0], [nd_cov], width=0.35, color=col30[0], edgecolor='black')

    #--ROC plot options
    axes[0].plot([0, 1], [0, 1], color='black', linestyle="--")
    axes[0].grid(linestyle="-.", color='black', lw=0.3)
    axes[0].set_ylim(-0.003, 1.003)
    axes[0].set_xlim(-0.003, 1.003)
    axes[0].set_xticks(np.arange(0,1.2,0.2))
    axes[0].set_xticklabels([0,0.2,0.4,0.6,0.8,1])
    axes[0].set_yticks(np.arange(0,1.2,0.2))
    axes[0].set_yticklabels([0,0.2,0.4,0.6,0.8,1])
    # plt.xlabel("False positive rate")
    # plt.ylabel("True positive rate")
    axes[0].set_title(title,y=1.02)

    #--Coverages plot options
    axes[1].grid(linestyle="-.", color='black', lw=0.3, axis='y')
    axes[1].set_xlim(-0.2,0.2)
    axes[1].set_ylim(-0.003, 1.003)
    #axes[1].set_title('Cov.')
    axes[1].set_ylabel('')
    axes[1].set_xlabel('')
    axes[1].set_xticks([])
    axes[1].set_yticks(np.arange(0,1.1,0.1))
    axes[1].set_yticklabels([])
    axes[1].yaxis.tick_right()
    axes[1].tick_params(axis='y', which='both',right=False)

    #--Legend plot options
    if len(dts) > 0:
        all_auc = [float(y) for x in aurocs.values() for y in x.split('|')]
        if max(all_auc) < 0.7 and min(all_auc) < 0.4:
            labels, handles = zip(*sorted(zip(*axes[0].get_legend_handles_labels()),
                                          key=lambda t: max([float(i) for i in t[1].split()[0].strip().split('|')]), reverse=True))
            lg =axes[0].legend(loc=2,frameon=False, bbox_to_anchor=(-0.03,1.03), handlelength=0.8)
        else:
            labels, handles = zip(*sorted(zip(*axes[0].get_legend_handles_labels()),
                                          key=lambda t: max([float(i) for i in t[1].split()[0].strip().split('|')]), reverse=True))
            axes[0].legend(labels, handles, loc=4, frameon=False, bbox_to_anchor=(1.04,-0.05), handlelength=0.8)

    plt.savefig(opath+'/roc_validation.png', bbox_inches='tight')
    plt.close()

    return aurocs

#------------------
if  __name__ == "__main__":

    #Verifying argments
    mpath = sys.argv[1]
    if len(sys.argv) > 2:
        ref_dt = sys.argv[2]
    else:
        ref_dt = None

    if len(sys.argv) >3:
        emb_path = sys.argv[3]
    else:
        emb_path = emb_path

    p = emb_path+'/%s/%s/'%(mpath[:3], mpath)
    if not os.path.exists(p):
        sys.exit('The metapath "%s" does not exist! Exiting...'%mpath)

    #Getting datasets (only if they have embeddings)
    dts = []
    for dt in os.listdir(p):
        if not os.path.exists(p+'/%s/nd2st.h5'%dt):
            sys.stderr.write('--> WARNING: Error when reading the embedding: %s|%s (nd2st.h5 not found). Skipping this dataset\n'%(mpath,dt))
        elif not os.path.exists(p+'/%s/edges.h5'%dt):
            sys.stderr.write('--> WARNING: Error when reading the embedding: %s|%s (edges.h5 not found). Skipping this dataset\n'%(mpath,dt))
        else:
            dts.append(dt)

    if ref_dt and ref_dt not in dts:
        sys.exit('The dataset "%s" is not found in the metapath "%s". Exiting...\n'%(ref_dt, mpath))

    #--Making directories
    for dt in dts:
        intra_path = p+'/%s'%dt
        for ñ in ['validation','w5','cross_charact','intra']:
            intra_path+='/%s'%ñ
            if not os.path.exists(intra_path):
                try:
                    os.mkdir(intra_path)
                except FileExistsError: #Sometimes the previous step fails when runing different embeddings at the same time
                    pass

    #Deciding if the metapath should be considered as directed or undirected (onlly important for the negative edges)
    undirected_metaedges = get_undirected_megs()

    if mpath[:3] == mpath[-3:]:
        if len(mpath.split('-')) == 3 and mpath not in undirected_metaedges:
            is_undirected = False
        else:
            is_undirected = True
    else:
        is_undirected = False

    for dt in tqdm(dts, desc='Datasets'):
        #--Variables
        intra_path = p+'/%s/validation/w5/cross_charact/intra/'%dt
        _args = parse_parameter_file(p+'/%s/parameters.txt'%dt)
        mnd1 = _args['source'] if _args['tag_source'] is None else _args['tag_source'] + _args['source']
        mnd2 = _args['target'] if _args['tag_target'] is None else _args['tag_target'] + _args['target']
        emb = get_emb(mpath, dt, [mnd1,mnd2], emb_path=emb_path)
        if mnd1 == mnd2:
            is_undirected = False if len(mpath.split('-')) == 3 and mpath not in undirected_metaedges else True
        else:
            is_undirected = False

        #--Iterating through the rest of datasets
        for dt2 in tqdm(dts, desc='%s'%dt, leave=False):
            if ref_dt and ref_dt not in [dt, dt2]: continue

            sub_dt_path = intra_path+'/%s/'%dt2
            if not os.path.exists(sub_dt_path):
                os.mkdir(sub_dt_path)
            #if os.path.exists(sub_dt_path+'/roc_validation.png'): continue

            #Getting validation edges

            for model_name, edges, cov_uv, cov_nodes_val, n_nodes_val, cov_edges_val, n_edges_val in tqdm(get_val_edges(p+'/%s/'%dt2, universe=set(emb)), leave=False, desc='Getting val edges'):
                if cov_uv == 0: continue

                #--Getting negative edges
                neg_edges =  get_neg_edges_from_egs_simple(edges, is_undirected=is_undirected, disable_tqdm=True)

                #Transforming edges to distances
                for dist_fn, nm in [(cosine, 'cos'), (euclidean, 'euc')]:
                    p_dist = [-dist_fn(emb[e[0]], emb[e[1]]) for e in edges]
                    n_dist = [-dist_fn(emb[e[0]], emb[e[1]]) for e in neg_edges]

                    if len(p_dist) == 0 or len(n_dist) == 0:break

                    #Calculating ROC curve
                    fpr,tpr,thr = roc_curve([1]*len(p_dist) + [0]*len(n_dist), p_dist+n_dist)
                    with open(sub_dt_path+'_roc_%s.%s'%(model_name, nm), 'w') as o:
                        o.write('\t'.join([str(x) for x in fpr])+'\n')
                        o.write('\t'.join([str(x) for x in tpr])+'\n')
                    del p_dist,n_dist, fpr,tpr,thr

            if cov_uv == 0: continue

            #Getting ROC curve
            title = '%s'%(dt2)
            if len(title) > 30:
                title = '\n'.join([n[:30]+'...' if len(n)>30 else n for n in title.split('-')])

            aurocs = plot_roc(sub_dt_path, sub_dt_path, nd_cov=cov_uv, title=title, figsize=(4.2,4), dpi=100)

            #Updating summary table
            summary_table = intra_path+'/results.csv'
            if len(aurocs) == 1:
                cols = ['cos.auroc', 'euc.auroc', 'nodes_cov', 'val_nodes_cov', 'val_nodes', 'val_edges_cov', 'val_edges']
            else:
                cols = ['cos.auroc_max','cos.auroc_25%','cos.auroc_50%','cos.auroc_100%',
                        'euc.auroc_max','euc.auroc_25%','euc.auroc_50%','euc.auroc_100%',
                        'nodes_cov', 'val_nodes_cov', 'val_nodes', 'val_edges_cov', 'val_edges']

            if os.path.exists(summary_table):
                df = pd.read_csv(summary_table, index_col=0, dtype=str).T
                #-------TEMPORAL REMOVE THIS AFTER UPDATING THE FILES!---------
                if 'val_nodes_cov' not in df.index.values:
                    df = pd.DataFrame(index=cols)
                #--------------------------------------------------------------
            else:
                df = pd.DataFrame(index=cols)

            if len(aurocs) == 1:
                df[dt2] = ['%s'%aurocs[''].split('|')[0], '%s'%aurocs[''].split('|')[1], '%.6f'%cov_uv, '%.6f'%cov_nodes_val, '%i'%n_nodes_val, '%.6f'%cov_edges_val, '%i'%n_edges_val]
                df = df[df.columns[list(np.argsort([max([float(y) for y in x.split('|')]) for x in df.loc['cos.auroc']])[::-1])]].T
            else:
                df[dt2] = ['%s'%aurocs['w-max'].split('|')[0], '%s'%aurocs['w-25%'].split('|')[0],'%s'%aurocs['w-50%'].split('|')[0],'%s'%aurocs['w-100%'].split('|')[0],
                           '%s'%aurocs['w-max'].split('|')[1], '%s'%aurocs['w-25%'].split('|')[1],'%s'%aurocs['w-50%'].split('|')[1],'%s'%aurocs['w-100%'].split('|')[1],
                           '%.6f'%cov_uv, '%.6f'%cov_nodes_val, '%i'%n_nodes_val, '%.6f'%cov_edges_val, '%i'%n_edges_val]
                df = df[df.columns[list(np.argsort([max([float(y) for y in x.split('|')]) for x in df.loc['cos.auroc_max']])[::-1])]].T

            df.to_csv(summary_table)
