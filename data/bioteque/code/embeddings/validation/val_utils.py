root_path = '../../../'
code_path = root_path+'code/embeddings/'
emb_path = root_path+'embeddings/'

import os
import sys
from tqdm import tqdm
from collections import Counter
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
try:
    from MulticoreTSNE import MulticoreTSNE as TSNE
except:
    from sklearn.manifold import TSNE

sys.path.insert(1, code_path)
from utils.graph_preprocessing import  read_edges
from utils.embedding_utils import get_embedding_universe, read_embedding, get_faiss_index

#Fontname
fname ='Arial'
#FontSize
fsize = 14

#FONT
font = {'family' : fname,
        'size'   : fsize }

plt.rc('font', **font)

plt.rcParams['pdf.fonttype'] = 42
sns.set(font_scale=1.25,font="Arial")
sns.set_style("whitegrid")
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})


col30 = [
    '#e6194B', '#4363d8', '#3cb44b', '#f58231','#911eb4',
    '#ffe119', '#46f0f0', '#f032e6', '#bcf60c', '#800000',
    '#008080', '#000075', '#e6beff', '#aaffc3', '#808000',
    '#00aaff', '#e55c00', '#66381a', '#606cbf', '#88ff00',
    '#391242', '#b38E50', '#985396', '#FACF63', '#58574b',
    '#9A6324', '#a9a9a9', '#fabebe', '#ffd8b1', '#fffac8',
]


def tsne_val(emb, opath1, opath2=None, nd2degree= None, nd2col=None, legend_handles=None, title='', dpi=100, alpha=0.7, dot_size=12):

    #Getting X,Y coordinates
    keys = np.asarray(list(emb.keys()))
    trans = TSNE(n_jobs=-1)
    node_embeddings_2d = trans.fit_transform(np.array([emb[k] for k in keys]))
    x,y = node_embeddings_2d[:,0], node_embeddings_2d[:,1]
    del emb

    #1) Ploting default colors
    if nd2col:
        colors = np.asarray([nd2col[k] for k in keys])

        #Sort the points by color count, so that the less count are plotted last
        col2count = Counter(colors)
        idx = np.argsort([col2count[c] for c in colors])[::-1]

    else:
        from scipy.stats import gaussian_kde
        xy = np.vstack([x, y])
        colors = gaussian_kde(xy)(xy)

        # Sort the points by density, so that the densest points are plotted last
        idx = colors.argsort()

    x, y, colors = x[idx], y[idx], colors[idx]
    keys = keys[idx]

    fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)

    g = ax.scatter(x,
                y,
                c = colors,
                s=dot_size,
                alpha=alpha,
                cmap='viridis',
                linewidths=0.05,
                edgecolor='black')
    ax.set_yticks([])
    ax.set_xticks([])

    if legend_handles is not None:
        ax.legend(handles=legend_handles, loc=9, frameon=False, handletextpad=0.001, ncol=len(legend_handles),
                   columnspacing=0.04, bbox_to_anchor=(0.5,1.13))
    else:
         ax.set_title(title)

    plt.savefig(opath1)
    plt.close()

    #2) Ploting by degree
    if nd2degree:

        colors = [np.log10(nd2degree[k]) for k in keys]
        fig, ax = plt.subplots(figsize=(4.7, 4), dpi=dpi)

        g = ax.scatter(x,
                    y,
                    c = colors,
                    s=dot_size,
                    alpha=alpha,
                    cmap='cool',
                    linewidths=0.05,
                    edgecolor='black')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(title)

        if len(set(colors)) < 6:
            ticks = np.unique(colors)
        else:
            ticks = list(np.round(np.arange(min(colors),max(colors), (max(colors)-min(colors))/5), 2))
            ticks[0] = min(colors)
            ticks.append(max(colors))

        cbar = fig.colorbar(g, ticks=ticks, pad=0)
        cbar.ax.set_yticklabels(['%i'%(round(10**x)) for x in ticks])  # horizontal colorbar

        if opath2:
            plt.savefig(opath2)
        plt.close()

def get_pos_neg_edges_for_node(edges, is_undirected=False, sep='\t',n_neigh=0.05, clip=(1,5), map_nd=None,bootstrap_neg=None):

    n1_to_neigh = {}
    n1_to_w = {}
    n2_to_neigh = {}
    n2_to_w = {}
    n1s = set([])
    n2s = set([])
    is_weighted = False
    if type(edges) == str:
        edges = read_edges(edges)

    #1) Reading neighboorhod and weights
    for h in edges:
        if map_nd:
            if type(map_nd) == dict:
                h[:2] = [map_nd[x] for x in h[:2]]
            else:
                h[:2] = map(map_nd,h[:2])

        if len(h) >2:
            is_weighted = True

        #--source node (n1)
        n1s.add(h[0])
        if h[0] not in n1_to_neigh:
            n1_to_neigh[h[0]] = []
            n1_to_w[h[0]] = []
        n1_to_neigh[h[0]].append(h[1])

        try:
            n1_to_w[h[0]].append(float(h[2]))
        except IndexError:
            n1_to_w[h[0]].append(1)

        #--target node (n2)
        if is_undirected:
            n1s.add(h[1])
            if h[1] not in n1_to_neigh:
                n1_to_neigh[h[1]] = []
                n1_to_w[h[1]] = []
            n1_to_neigh[h[1]].append(h[0])

            try:
                n1_to_w[h[1]].append(float(h[2]))
            except IndexError:
                n1_to_w[h[1]].append(1)

        else:
            n2s.add(h[1])
            if h[1] not in n2_to_neigh:
                n2_to_neigh[h[1]] = []
                n2_to_w[h[1]] = []
            n2_to_neigh[h[1]].append(h[0])
            try:
                n2_to_w[h[1]].append(float(h[2]))
            except IndexError:
                n2_to_w[h[1]].append(1)

    if is_weighted:
        sys.stderr.write('\n ** Weights detected. Using weights as prob. for sampling...\n')

    #2) Getting pos/neg samples for each node
    if is_undirected:
        ll = [(n1_to_neigh,n1_to_w,n1s)]
    else:
        ll = [(n1_to_neigh,n1_to_w,n2s),(n2_to_neigh,n2_to_w, n1s)]


    for n_to_neigh, n_to_w, neigh_uv in ll:
        n_to_splits = {}

        for node in n_to_neigh:

            if type(n_neigh) == float:
                N = np.clip(round(len(n_to_neigh[node])*n_neigh), clip[0], clip[1])
            else:
                N = np.clip(n_neigh, clip[0],clip[1])

            N = min([N, len(n_to_neigh[node])])


            #If weighted
            if is_weighted:

                n_to_neigh[node] = np.asarray(n_to_neigh[node])
                n_to_w[node] = np.asarray(n_to_w[node])

                sorted_w_ix = np.argsort(n_to_w[node])[::-1]
                pos = np.asarray(n_to_neigh[node])[sorted_w_ix[:N]]

                #NO-neigh
                #--Although I only get N neighbours I remove those neighbours with same weight that the N neighbour
                all_pos = set(n_to_neigh[node][np.where(n_to_w[node] >= n_to_w[node][sorted_w_ix][N-1])])
                no_neigh = list(neigh_uv.difference(all_pos))

                if len(no_neigh) > 0:
                    if type(bootstrap_neg) == int:
                        neg = [np.random.choice(no_neigh, len(pos), replace=True) for _ in range(bootstrap_neg)]
                    else:
                        neg = np.random.choice(no_neigh, len(pos), replace=True)
                else:
                    neg = []

            #If NOT weighted
            else:
                neigh = n_to_neigh[node]
                no_neigh = list(neigh_uv.difference(set(n_to_neigh[node])))
                assert len(set(neigh)&set(no_neigh)) == 0

                #Getting positive
                pos = np.random.choice(neigh, N, replace=False)

                #Getting negative
                if type(bootstrap_neg) == int:
                    neg = [np.random.choice(no_neigh,len(pos), replace=True) for _ in range(bootstrap_neg)]

                else:
                    neg = np.random.choice(no_neigh,len(pos), replace=True)

            yield node,pos,neg

def get_labels_dist_rank(mp,dt,distance='euclidean',max_nneigh=0.5,mnd1=None,mnd2=None,w=5,pp=False, is_undirected=False, emb_path = emb_path):
    from embedding_distances import get_nn_from_faiss
    if mnd1 == None:
        mnd1 = mp[:3]
    if mnd2 == None:
        mnd2 = mp[-3:]

    lbs1 = get_embedding_universe(mpath=mp, dt=dt,mnd=mnd1, emb_path=emb_path)
    lbs2 = get_embedding_universe(mpath=mp, dt=dt,mnd=mnd2, emb_path=emb_path)

    #rk1
    m1 = read_embedding(mpath=mp, dt=dt, w=w, pp=pp, mnd= mnd1, just_the_matrix=True, emb_path=emb_path)
    faiss_ix1 =  get_faiss_index(mpath=mp, dt=dt, w=w, pp=pp, mnd=mnd2, distance=distance, emb_path=emb_path)
    if max_nneigh == 'all' or max_nneigh==1:
        K = len(lbs2)
    elif type(max_nneigh) == float:
        K = int(len(lbs2)*max_nneigh)
    ds1, rk1 = get_nn_from_faiss(faiss_ix1, m1, k=K)

    if is_undirected:
        return lbs1, ds1, rk1
    else:

        #rk2
        m2 = read_embedding(mpath=mp, dt=dt, w=w, pp=pp, mnd= mnd2, just_the_matrix=True, emb_path= emb_path)
        faiss_ix2 =  get_faiss_index(mpath=mp, dt=dt, w=w, pp=pp, mnd=mnd1, distance=distance, emb_path=emb_path)
        if max_nneigh == 'all' or max_nneigh==1:
            K = len(lbs1)
        elif type(max_nneigh) == float:
            K = int(len(lbs1)*max_nneigh)
        ds2, rk2 = get_nn_from_faiss(faiss_ix2, m2, k=K)

        return lbs1,ds1,rk1,lbs2,ds2,rk2
