root_path = '../../../'
code_path = root_path+'code/embeddings/'
emb_path = root_path+'embeddings/'

import sys
import os
import numpy as np
import pandas as pd
import h5py
import shutil
import pickle
from collections import Counter
from tqdm import tqdm
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics import roc_auc_score, roc_curve
try:
    from MulticoreTSNE import MulticoreTSNE as mTSNE
except:
    from sklearn.manifold import TSNE as mTSNE
from sklearn.manifold import TSNE

sys.path.insert(1, code_path)
from utils.graph_preprocessing import read_edges, get_neg_edges_from_egs_simple
from utils.embedding_utils import read_embedding
from utils.utils import get_undirected_megs
from validation.val_utils import  col30

#from reference_vals import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import gridspec, ticker
import seaborn as sns

fsize = 12
fname = 'arial'

def get_image_and_dist(emb_path, order=None,  image_ixs = 100, dist_n_pairs=10000, shuffle=True):

    if order is None:
        paths = [emb_path+'/%s'%f for f in sorted(os.listdir(emb_path))]
    else:
        paths = [emb_path+'/%s.h5'%mnd for mnd in order]

    if len(paths) == 1:
        is_homogeneous = True
    else:
        is_homogeneous = False

    m = []
    euclidean_dist = []
    cosine_dist = []
    emb = [[],[]]
    mnds_names = []
    for I in range(len(paths)):
        p = paths[I]
        mnd_name = p.split('/')[-1][:-3]
        mnds_names.append(mnd_name)

        with h5py.File(p, 'r') as f:
            M = f['m'][:]

            ixs = np.arange(M.shape[0])
            if shuffle:
                ixs = np.random.permutation(ixs)
                ixs = sorted(ixs[:image_ixs])
            else:
                ixs = [i for i in ixs if i>= 0 and i < M.shape[0]]
            m.append(M[ixs,:])

            #Duplicate the Image with other random ixs
            if is_homogeneous:
                ixs = np.arange(M.shape[0])
                if shuffle:
                    ixs = np.random.permutation(ixs)
                    ixs = sorted(ixs[:image_ixs])
                else:
                    ixs = [i for i in ixs if i>= 0 and i < M.shape[0]]
                m.append(M[ixs,:])

        #--Metanode Distances
        euc_dist, cos_dist = [],[]
        for _ in range(dist_n_pairs):
            i, j = np.random.choice(M.shape[0], 2, replace=False)
            euc_dist.append(euclidean(M[i], M[j]))
            cos_dist.append(cosine(M[i], M[j]))
            emb[I].append(M[i])

        euclidean_dist.append((mnd_name, euc_dist))
        cosine_dist.append((mnd_name, cos_dist))

    #--Edge Distances
    if not is_homogeneous:
        euclidean_dist.append(('-'.join(mnds_names), [euclidean(emb[0][i], emb[1][i]) for i in range(dist_n_pairs)]))
        cosine_dist.append(('-'.join(mnds_names), [cosine(emb[0][i], emb[1][i]) for i in range(dist_n_pairs)]))

    return m, euclidean_dist, cosine_dist

def get_tsne_data(emb, nd2col=None, nd2degree= None, multicore=False):

    keys = np.asarray(list(emb.keys()))

    trans = mTSNE(n_jobs=-1) if multicore is True else TSNE(n_components=2)

    node_embeddings_2d = trans.fit_transform(np.array([emb[k] for k in keys]))
    x,y = node_embeddings_2d[:,0], node_embeddings_2d[:,1]
    #del emb

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

    return {'x':x,'y':y,'keys':keys,'colors':colors}

def get_val_edges(path, universe=None, percentile_list = [('w-max',0), ('w-25%',0.25), ('w-50%',0.5), ('w-100%',1)]):

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

        for name,prc in percentile_list:
            edges = set([])
            for n in nd2ew:
                cutoff =nd2ew[n][max([0,int(len(nd2ew[n])*prc)-1]),1]
                gns = [id2nd[x] for x in nd2ew[n][np.where(nd2ew[n][:,1] >= cutoff)[0],0]]
                edges.update(zip([id2nd[n]]*len(gns), gns))

            yield  name, edges, cov_uv, cov_nodes_val, n_nodes_val, cov_edges_val, n_edges_val
    else:
        yield  '', set([(id2nd[e[0]], id2nd[e[1]]) for e in edges]) , cov_uv,  cov_nodes_val, n_nodes_val, cov_edges_val, n_edges_val


def write_sanity_check_dist(mnds, mp,dt,is_undirected, opath, dist_fn=cosine, permutations=10, percentile_list=[('w-max',0), ('w-25%',0.25), ('w-50%',0.5), ('w-100%',1)],
                           emb_path = emb_path):

    if not os.path.exists(opath):
        os.mkdir(opath)

    emb = read_embedding(mpath=mp, dt=dt, mnd=mnds[0], emb_path=emb_path)
    emb.update(read_embedding(mpath=mp, dt=dt, mnd=mnds[-1], emb_path=emb_path))

    with h5py.File(opath+'/dist.h5','w') as o:
        for r in tqdm(get_val_edges(emb_path+'/%s/%s/%s'%(mp[:3], mp, dt), universe=set(emb), percentile_list=percentile_list), desc='Models'):

            #Getting edges
            model, edges = r[0], r[1]

            #Getting positive distance and save it
            p_dist = np.asarray([dist_fn(emb[e[0]], emb[e[1]]) for e in edges])
            o.create_dataset(model+'_pos', data=p_dist)
            del p_dist

            #Getting neg edges distances
            n_dist = []
            for _ in tqdm(range(permutations), desc='Neg permutations', leave=False):
                neg_edges =  get_neg_edges_from_egs_simple(edges, is_undirected=is_undirected, disable_tqdm=True)
                n_dist.append(np.asarray([dist_fn(emb[e[0]], emb[e[1]]) for e in neg_edges]))

            #Save neg distances
            o.create_dataset(model+'_neg', data=np.array(n_dist))

def get_recovery_roc_plot(opath, models=['w-max', 'w-25%', 'w-50%', 'w-100%']):
    #models = ['w-max', 'w-25%', 'w-50%', 'w-100%']
    #model2name = {'w-max':'TOP1', 'w-25%':'25%', 'w-50%':'50%', 'w-100%':'100%'}

    with h5py.File(opath, 'r') as f:
        _ = []

        for ix, model in enumerate(models):
            if model+'_pos' not in f.keys(): continue
            pos = f[model+'_pos'][:]
            neg = []
            for r in f[model+'_neg'][:]:
                neg.extend(r)

            pred = list(-pos)+list(-np.array(neg))
            truth = [1]*len(pos)+[0]*len(neg)

            fpr, tpr, thr = roc_curve(truth,pred)

            #ax.plot(fpr,tpr, color=col30[ix], label=)
            _.append({'fpr':fpr, 'tpr':tpr, 'color':col30[ix], 'auroc':'%.2f'%(roc_auc_score(truth,pred))})

    return _

def get_inter_charact(mp,dt, uvs=None,max_meg=10, min_cov=0.1, val2name=None, auroc_col='cos.auroc', emb_path=emb_path):

    def read_inter_df(mp, dt, emb_path):
        return pd.read_csv(emb_path+'/%s/%s/%s/validation/w5/cross_charact/inter/results.csv'%(mp[:3], mp, dt), index_col=0)

    try:
        k = read_inter_df(mp,dt, emb_path)
    except FileNotFoundError:
        return None, None

    vals_uv = set(k.index.values)

    m, m_cov = [], []
    labels = []

    if uvs is not None:
        megs_uv = uvs
        for _mp,_dt in megs_uv:
            if _mp not in vals_uv:
                m.append(np.nan)
            else:
                m.append(float(k[k['dt']==_dt].loc[_mp][auroc_col]))
                m_cov.append(float(k[k['dt']==_dt].loc[_mp]['nodes_cov']))


            if val2name and (_mp,_dt) in val2name:
                labels.append(val2name[(_mp,_dt)])
            else:
                labels.append(_mp)
    else:
        already = set([])
        for _mp,r in k.iterrows():
            _dt = r['dt']
            if _mp in already: continue
            auroc = r[auroc_col]
            cov = r['nodes_cov']
            if min_cov and cov < min_cov: continue

            #if auroc < 0.6 and len(m)>= min_meg: break
            m.append(auroc)
            m_cov.append(cov)
            already.add(_mp)
            if val2name and (_mp,_dt) in val2name:
                labels.append(val2name[(_mp,_dt)])
            else:
                labels.append(_mp)
            if len(m) >= max_meg: break

        df_auroc = pd.DataFrame([m], columns=labels)
        df_cov = pd.DataFrame([m_cov], columns=labels)

    return df_auroc, df_cov


def get_mnd_charact(mnds, mp,dt, uvs=None, min_cov = 0.1, max_meg = 10,val2name=None, auroc_col = 'cos.auroc', emb_path=emb_path):

    def read_mnd_df(mp, dt, mnd, emb_path):
        return pd.read_csv(emb_path+'/%s/%s/%s/validation/w5/cross_charact/mnd/%s/results.csv'%(mp[:3], mp, dt, mnd), index_col=0)

    dfs_auroc, dfs_cov = [], []
    for i in range(len(set(mnds))):
        mnd = mnds[i]

        try:
            k = read_mnd_df(mp, dt, mnd=mnd, emb_path=emb_path)
        except FileNotFoundError:
            dfs_auroc.append(None)
            dfs_cov.append(None)
            continue

        vals_uv = set(k.index.values)
        m = []
        m_cov = []
        labels = []

        if uvs is not None:
            megs_uv = uvs[i]
            for _mp,_dt in megs_uv:
                if _mp not in vals_uv:
                    m.append(np.nan)
                else:
                    m.append(float(k[k['dt']==_dt].loc[_mp][auroc_col]))
                    m_cov.append(float(k[k['dt']==_dt].loc[_mp]['nodes_cov']))

                if val2name and (_mp,_dt) in val2name:
                    labels.append(val2name[(_mp,_dt)])
                else:
                    labels.append(_mp)
        else:
            already = set([])
            for _mp,r in k.iterrows():

                _dt = r['dt']
                if _mp in already: continue
                auroc = r[auroc_col]
                cov = r['nodes_cov']
                if min_cov and cov < min_cov: continue

                #if auroc < 0.6 and len(m)>= min_meg: break
                m.append(auroc)
                m_cov.append(cov)
                already.add(_mp)
                if val2name and (_mp,_dt) in val2name:
                    labels.append(val2name[(_mp,_dt)])
                else:
                    labels.append(_mp)
                if len(m) >= max_meg: break

        df = pd.DataFrame([m], columns=labels)
        df_cov = pd.DataFrame([m_cov], columns=labels)

        dfs_auroc.append(df)
        dfs_cov.append(df_cov)
    return dfs_auroc, dfs_cov



#---------------
current_path = sys.argv[1]

mp,dt = current_path.rstrip('/').split('/')[-2:]
emb_path = '/'+'/'.join([x for x in current_path.split('/') if x!=''][:-3]) +'/'

mnds = np.unique([mp[:3],mp[-3:]])
if len(mnds) == 1 and '__pg__%s.h5'%mnds[0] in os.listdir(current_path+'/embs/w5/'): #Checking if one comes from a PGN
    mnds = np.array([mnds[0], '__pg__%s'%mnds[0]])
mnd2name = dict([(mnd, 'p'+mnd.split('_')[-1]) if mnd.startswith('__pg__') else (mnd,mnd) for mnd in mnds])

colors = ['#5078dc','#fa6450']
mnd2col = dict(zip(mnds, colors))
scratch = current_path+'/scratch/emb_card/'
if not os.path.exists(scratch):
    os.mkdir(scratch)

undirected_metaedges = get_undirected_megs()
if mp[:3] == mp[-3:]:
    if len(mp.split('-')) == 3 and mp not in undirected_metaedges:
        is_undirected = False
    else:
        is_undirected = True
else:
    is_undirected = False

# 1) tSNE data
sys.stderr.write('1) Getting tSNE data...\n')
k2col = {}
#--ALL metanodes
if len(mnds) > 1:
    _e = {}

    #--n1
    n = read_embedding(current_path+'/embs/w5/%s.h5'%(mnds[0]))
    k2col.update(zip(list(n), [colors[0]]*len(n)))
    _e.update(n)

    #--n2
    n = read_embedding(current_path+'/embs/w5/%s.h5'%(mnds[1]))
    k2col.update(zip(list(n), [colors[1]]*len(n)))
    _e.update(n)

    del n

    #--Get emb
    _e.update(read_embedding(current_path+'/embs/w5/%s.h5'%(mnds[1])))

    #legend_handles = [Line2D([0], [0], marker='o', color=color,label='%s'%''.join([mnd2name]), markerfacecolor=color, markersize=7, linewidth=0)
    #               for mnd,color in mnd2col.items()] #tags borders (__) are removed as python don't allow them to be on the legend

    try:
        tsne_data = get_tsne_data(_e, nd2col = k2col, multicore = True)
    except RuntimeError:
        tsne_data = get_tsne_data(_e, nd2col = k2col, multicore = False)

    #tsne_data['legend'] = legend_handles

else:
    _e = read_embedding(current_path+'/embs/w5/%s.h5'%(mnds[0]))

    try:
        tsne_data = get_tsne_data(_e, nd2col = dict(zip(list(_e), [colors[0]]*len(_e))), multicore = True)
    except RuntimeError:
        tsne_data = get_tsne_data(_e, nd2col = dict(zip(list(_e), [colors[0]]*len(_e))), multicore = False)

# 2) Image and Dist
sys.stderr.write('2) Getting Image and dist data...\n')
images,euclideans,cosines = get_image_and_dist(current_path+'/embs/w5/', order=None,  dist_n_pairs=10000)

# 3) Intra val
sys.stderr.write('3) Getting ROC validation...\n')
permutations = 10
with h5py.File(current_path+'/edges.h5', 'r') as f:
    models = [('w-50%',0.5)] if 'weights' in f.keys() else [('', None)]

write_sanity_check_dist(mnds, mp,dt,is_undirected, scratch, dist_fn=cosine, permutations=permutations, percentile_list=models, emb_path=emb_path)
recovery_roc = get_recovery_roc_plot(scratch+'/dist.h5', models=[x[0] for x in models])

# 4) Lollipop plots
sys.stderr.write('4) Getting characterization data...\n')
mnd_charact_auroc, mnd_charact_cov = get_mnd_charact(mnds, mp,dt, min_cov = 0.1, emb_path=emb_path)
inter_charact_auroc, inter_charact_cov = get_inter_charact(mp,dt, min_cov = 0.1, emb_path=emb_path)

#----------------
#  Saving inputs
#----------------
with open(current_path+'/scratch/emb_card_input.pkl','wb') as o:
    r = {
        'tsne_data': tsne_data,
        'images': images,
        'euclideans': euclideans,
        'cosines': cosines,
        'recovery_roc': recovery_roc,
        'mnd_charact_auroc': mnd_charact_auroc,
        'mnd_charact_cov': mnd_charact_cov,
        'inter_charact_auroc': inter_charact_auroc,
        'inter_charact_cov': inter_charact_cov
        }
    pickle.dump(r,o)
#----------------

#--------------------------
#5) Plotting
#--------------------------
figsize=(12,8.5)
fig = plt.figure(figsize=figsize, dpi=100, constrained_layout=False)
#sum(np.array([0.4, 0.23, 0.37])* (5/9))
gs = gridspec.GridSpec(4, 3, width_ratios= [0.44, 0.25, 0.25], height_ratios=[0.23529412, 0.13529412, 0.21764706, 0.4117647058823529], figure=fig)
plt.subplots_adjust(hspace=0.5, wspace=0.18)

ax1 = fig.add_subplot(gs[:3, 0])
ax2_grid = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[:2,1], hspace=0.18)
ax2 = [fig.add_subplot(ax2_grid[0,:]), fig.add_subplot(ax2_grid[1,:])]
ax3_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2,1], hspace=0.1)
ax3 = [fig.add_subplot(ax3_grid[0]), fig.add_subplot(ax3_grid[1])]

ax4_grid = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[:2,2], hspace=0.18)
ax4 = [fig.add_subplot(ax4_grid[:4,:])]
ax5_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2,2], hspace=0.1)
ax5= [fig.add_subplot(ax5_grid[0]), fig.add_subplot(ax5_grid[1])]

if len(set(mnds))==1:
    ax6_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3,:], wspace=0.05)
    ax6= [fig.add_subplot(ax6_grid[0]), fig.add_subplot(ax6_grid[1])]
else:
    ax6_grid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[3,:], wspace=0.05)
    ax6= [fig.add_subplot(ax6_grid[0]), fig.add_subplot(ax6_grid[1]), fig.add_subplot(ax6_grid[2])]

#1) tSNE
x, y, colors, = tsne_data['x'], tsne_data['y'],tsne_data['colors']
g = ax1.scatter(x,
            y,
            c = colors,
            s=12,
            alpha=0.7,
            cmap='viridis',
            linewidths=0.02,
            edgecolor='black')
ax1.set_yticks([])
ax1.set_xticks([])
ax1.set_title(mp, fontsize=fsize-1, fontfamily=fname, y=1.005)

#2) Images

#--mnd1
ax2[0].imshow(images[0], cmap='coolwarm', aspect="auto")
ax2[0].set_yticks([])
#ax2[0].set_yticks(np.arange(0,100+int(100/4),int(100/4)))
ax2[0].set_xticks([0,31,63,95,127])
ax2[0].set_xticklabels([])
ax2[0].set_ylabel(mnd2name[mnds[0]], fontsize=fsize, fontfamily=fname)
ax2[0].set_title('Embedding', fontsize=fsize, fontfamily=fname)

#--mnd2
ax2[1].imshow(images[1], cmap='coolwarm', aspect="auto")
ax2[1].set_yticks([])
#ax2[1].set_yticks(np.arange(0,100+int(100/4),int(100/4)))
ax2[1].set_xticks([0,31,63,95,127])
ax2[1].set_xticklabels(['0','32','64','96','128'], fontsize=fsize, fontfamily=fname)
ax2[1].set_ylabel(mnd2name[mnds[-1]], fontsize=fsize, fontfamily=fname)

#3) Value dist
if len(set(mnds))==1:
    sns.histplot(np.concatenate(images).ravel(), ax=ax3[1], color=mnd2col[mnds[0]], kde=True, stat='density')
else:
    sns.histplot(images[0].ravel(), ax=ax3[1], color=mnd2col[mnds[0]], kde=True, stat='density')
    sns.histplot(images[1].ravel(), ax=ax3[1], color=mnd2col[mnds[-1]], kde=True, stat='density')

ax3[1].set_yticks([])
ax3[1].set_title('Values', fontsize=fsize, fontfamily=fname)
ax3[1].set_ylabel('Density', fontsize=fsize-1, fontfamily=fname)
for tick in ax3[1].xaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)

#4) Roc Curves
for ix,d in enumerate(recovery_roc):
    fpr,tpr,col, label = d['fpr'],d['tpr'], d['color'], d['auroc']

    ax4[0].plot(fpr,tpr, color=col, linestyle='-', label=label)

ax4[0].plot([0, 1], [0, 1], color='black', linestyle="--")
ax4[0].grid(linestyle="-.", color='black', lw=0.3)
ax4[0].set_ylim(-0.003, 1.003)
ax4[0].set_xlim(-0.003, 1.003)
ax4[0].set_xticks(np.arange(0,1.2,0.2))
ax4[0].set_xticklabels([0,0.2,0.4,0.6,0.8,1], fontsize=fsize, fontfamily=fname)
ax4[0].set_yticks(np.arange(0,1.2,0.2))
ax4[0].set_yticklabels([0,0.2,0.4,0.6,0.8,1], fontsize=fsize, fontfamily=fname)
#ax4[0].set_ylabel('TPR')
#ax4[0].set_xlabel('FPR')
ax4[0].set_title('Network preservation', fontsize=fsize, fontfamily=fname)
L = ax4[0].legend(frameon=True, handlelength=0.5, handletextpad=0.5, loc=4, title='AUROC')
plt.setp(L.texts, fontsize=fsize, family=fname)
plt.setp(L.get_title(),fontsize=fsize)

#5) Distances
ix = 0

for lb, d in euclideans:
    color = '#c864e1' if ix == 2 else mnd2col[mnds[ix]]
    sns.histplot(d, ax=ax5[0], color=color, kde=True, stat='density')
    ix+=1

ix = 0
for lb, d in cosines:
    color = '#c864e1' if ix == 2 else mnd2col[mnds[ix]]
    sns.histplot(d, ax=ax5[1], color=color, kde=True, stat='density')
    ix+=1

ax5[0].set_yticks([])
ax5[0].set_title('Euclidean', fontsize=fsize, fontfamily=fname)
ax5[1].set_yticks([])
ax5[1].set_title('Cosine', fontsize=fsize, fontfamily=fname)
ax5[0].xaxis.set_major_locator(ticker.MultipleLocator(2))
ax5[0].xaxis.set_major_formatter(ticker.ScalarFormatter())
ax5[0].set_ylabel('Density', fontsize=fsize-1, fontfamily=fname)
for tick in ax5[0].xaxis.get_major_ticks():
    tick.label.set_fontsize(fsize)

ax5[1].set_xticks([0,0.5,1])
ax5[1].set_xticklabels([0, 0.5, 1], fontsize=fsize, fontfamily=fname)
ax5[1].set_ylabel('')

#Edge charact
if inter_charact_auroc is not None:
    color = '#c864e1'

    X = np.arange(inter_charact_auroc.shape[1])
    adjust_dodge = [-0.25,0,0.25]

    nd_coverages = list(inter_charact_cov.iloc[0])
    for i in range(inter_charact_auroc.shape[0]):
        model = inter_charact_auroc.index.values[i]
        dodge = 0
        label = model
        ax6[0].scatter(X, list(inter_charact_auroc.iloc[i]), color=color,s=[50*s for s in nd_coverages], label=label, zorder=11)
        ax6[0].vlines(X, list(inter_charact_auroc.iloc[i]), 0.5, color=color, zorder=10)

    ax6[0].set_xticks(np.arange(len(X)))
    ax6[0].set_xticklabels(inter_charact_auroc.columns,rotation=50, ha='right', fontsize=fsize, fontfamily=fname)
    ax6[0].set_xlim(-0.5, inter_charact_auroc.shape[1]-0.5)
else:
    ax6[0].set_xticks([])

ax6[0].set_title('-'.join([mnd2name[mnds[0]],mnd2name[mnds[-1]]]), fontsize=fsize, fontfamily=fname)
ax6[0].set_yticks([0.5,0.6,0.7,0.8,0.9,1])
ax6[0].set_yticklabels([0.5,0.6,0.7,0.8,0.9,1], fontsize=fsize, fontfamily=fname)
ax6[0].grid(axis='y', linestyle='--', alpha=0.8)
ax6[0].set_ylim(0.5,1.0)
ax6[0].set_ylabel('AUROC', fontsize=fsize, fontfamily=fname)

#Metanode charact
for ix, df in enumerate(mnd_charact_auroc):
    ax = ax6[ix+1]

    if df is not None:
        color = mnd2col[mnds[ix]]
        nd_coverages = list(mnd_charact_cov[ix].iloc[0])
        X = np.arange(df.shape[1])
        adjust_dodge = [-0.25,0,0.25]
        for i in range(df.shape[0]):
            model = df.index.values[i]
            dodge = 0
            label = model
            ax.scatter(X, list(df.iloc[i]), color=color,s=[50*s for s in nd_coverages], label=label, zorder=11)
            ax.vlines(X, list(df.iloc[i]), 0.5, color=color, zorder=10)

        ax.set_yticks([0.5,0.6,0.7,0.8,0.9,1])
        ax.set_yticklabels([0.5,0.6,0.7,0.8,0.9,1], fontsize=fsize, fontfamily=fname)
        ax.set_xticks(np.arange(len(X)))
        ax.set_xticklabels(df.columns,rotation=50, ha='right',fontsize=fsize, fontfamily=fname)
        ax.set_xlim(-0.5, df.shape[1]-0.5)

    else:
        ax.set_xticks([])

    ax.grid(axis='y', linestyle='--', alpha=0.8)
    ax.set_ylim(0.5,1.0)
    ax.set_title(mnd2name[mnds[ix]], fontsize=fsize, fontfamily=fname)

ax6[1].set_yticklabels([])
if mnds[0] != mnds[-1]:
    ax6[2].set_yticklabels([])

#Legend
ax3[0].remove()

legend_handles = [Line2D([0], [0], marker='o', color=mnd2col[mnd],label='%s'%mnd2name[mnd], markerfacecolor=mnd2col[mnd], markersize=7, linewidth=0)
                   for mnd in mnds]
legend_handles.append(Line2D([0], [0], marker='o', color='#c864e1',label='%s-%s'%(mnd2name[mnds[0]], mnd2name[mnds[-1]]), markerfacecolor='#c864e1', markersize=7, linewidth=0))
L = ax1.legend(handles=legend_handles, loc=7, frameon=False, handletextpad=-0.5, ncol=1, bbox_to_anchor=(1.40,0.14))
plt.setp(L.texts, family=fname,size=fsize,color='black',alpha=1)

#Saving figure
plt.savefig(current_path+'/emb_card.png', dpi=300, bbox_inches='tight')

#Cleaning scratch
shutil.rmtree(scratch)
