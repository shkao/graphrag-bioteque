root_path = '../../../'
code_path = root_path+'code/embeddings/'
emb_path = root_path+'embeddings/'
singularity_image = root_path+'/programs/ubuntu-py.img'
hpc_path = root_path+'/programs/hpc/'

import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import gmean
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import shutil
import pickle
import argparse
sys.path.insert(1, code_path)
from utils import graph_preprocessing as gpr
from utils.utils import parse_parameter_file
from validation.val_utils import *
sys.path.insert(0, hpc_path)
from hpc import HPC
from config import config as cluster_config


## Variables
def get_parser():
    description = 'Runs metapath2vec'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-path', '--path',  type=str, required=True, help='Path of the embedding folder "current_path"')
    parser.add_argument('-d', '--distance', type=str, required=False, default='euclidean', help="Distance to be used (default=euclidean)")
    parser.add_argument('-pp', '--pp',  type=int, required=False, default=0, help='If 1 it uses matapath2vec++ embeddings (default=0)')
    parser.add_argument('-w', '--window', type=int, required=False, default=5, help='Window size used (default=5)')
    parser.add_argument('-r', '--max_rk', type=float, required=False, default=0.75, help="Max p-value precision (default=0.5)")
    parser.add_argument('-mnneigh', '--mnneigh', type=int, required=False, default=30, help='Min number of neighbours considerd (default=30)')
    parser.add_argument('-mxneigh', '--mxneigh', type=float, required=False, default=0.1, help='Max number of neighbours considered (default=0.1)')
    parser.add_argument('-b', '--n_bootstrap', type=int, required=False, default=10, help='Number of neg bootstrap to perform (default=10)')
    return parser

args = get_parser().parse_args(sys.argv[1:])
current_path = args.path
distance = args.distance
pp = bool(args.pp)
window = args.window
pvalue_precision = args.max_rk
min_split_neigh = args.mnneigh #Min number of neighboors to take (if available) for each node in the split. This prevents that taking too few neighboors if 0.1 is small
max_split_neigh = args.mxneigh #Proportion of neighboors to take for each node in the split
bootstrap_neg = args.n_bootstrap
emb_path = '/'+'/'.join([x for x in current_path.split('/') if x!=''][:-3]) +'/'

is_weighted = False #It will change to True if weights are detected in the edges
split_nodes = 100 #Elements in the cluster
if distance not in {'euclidean','cosine'}:
    sys.exit('Invalid distance: %s'%distance)

#making directories and setting scratch and output path

#--validation path
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
#--scratch_path
scratch_path = current_path+'/scratch/'
if not os.path.exists(scratch_path):
    os.mkdir(scratch_path)
scratch_path = current_path+'/scratch/validation/'
if not os.path.exists(scratch_path):
    os.mkdir(scratch_path)
scratch_path = current_path+'/scratch/validation/%s/'%distance
if not os.path.exists(scratch_path):
    os.mkdir(scratch_path)


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

if mnd1 != mnd2 or (mnd1 == mnd2 and len(mp.split('-'))==3 and mp not in gpr._undirected_megs):
    is_undirected=False
else:
    is_undirected=True

#Getting labels, distances and ranks
if is_undirected:
    lbs1,ds1,rk1 = get_labels_dist_rank(mp, dt, mnd1=mnd1, mnd2=mnd2, distance=distance, max_nneigh=pvalue_precision, w= window, pp=pp, is_undirected=is_undirected, emb_path=emb_path)
    lbs1 = np.asarray([nd2id[n] for n in lbs1])
    lbs2 = lbs1
    rk2 = rk1
    ds2 = ds1
else:
    lbs1,ds1,rk1,lbs2,ds2,rk2 = get_labels_dist_rank(mp, dt, mnd1=mnd1, mnd2=mnd2, distance=distance, max_nneigh=pvalue_precision, w= window, pp=pp, is_undirected=is_undirected, emb_path=emb_path)
    lbs1 = np.asarray([nd2id[n] for n in lbs1])
    lbs2 = np.asarray([nd2id[n] for n in lbs2])

lbs = [lbs1,lbs2]
#Kepping into the scratch
with h5py.File(scratch_path+'/rks.h5','w') as o:
    o.create_dataset('rk1',data=rk1)
    o.create_dataset('rk2',data=rk2)

with h5py.File(scratch_path+'/dss.h5','w') as o:
    o.create_dataset('ds1',data=ds1)
    o.create_dataset('ds2',data=ds2)

with open(scratch_path+'/lbs.pkl','wb') as o:
    pickle.dump(lbs,o)

lbs1 = set(lbs1)
lbs2 = set(lbs2)

all_nodes = set([])
with open(scratch_path+'/n2split.tsv','w') as o:
    for  n, pos, negs in tqdm(get_pos_neg_edges_for_node(current_path+'/edges.h5', n_neigh=max_split_neigh, clip=(1,5),
                                                        is_undirected=is_undirected, bootstrap_neg=bootstrap_neg, map_nd=str),desc='Writing splits'):
        all_nodes.add(n)
        if n in lbs1:
            position = 0
        else:
            position = 1

        if type(negs[0]) != list and type(negs[0]) != np.ndarray:
            negs = [negs]
        o.write('%i\t%s\t%s\t%s\n'%(position,n,','.join(pos),'|'.join([','.join(x) for x in negs])))

#------- RUN CODE IN THE CLUSTER  -----------------------
sys.stderr.write('Running ROC curve calculation in the cluster. Bootstrap neg. --> %i\n'%bootstrap_neg)
sys.stderr.flush()
#Making folders
clust_scratch = scratch_path+'/_cluster/'
if not os.path.exists(clust_scratch):
    os.mkdir(clust_scratch)
if not os.path.exists(clust_scratch+'/_nd2AUC'):
    os.mkdir(clust_scratch+'/_nd2AUC')
if not os.path.exists(clust_scratch+'/_all_preds'):
    os.mkdir(clust_scratch+'/_all_preds')
tmp_script="""
import sys
import uuid
from tqdm import tqdm
import pickle
import h5py
import numpy as np
from scipy.stats import gmean
from sklearn.metrics import roc_auc_score

task_id = sys.argv[1]  # <TASK_ID>
filename = sys.argv[2]  # <FILE>
inputs = pickle.load(open(filename, 'rb'))  # load
scratch_path,distance,nodes_uvs = inputs[task_id][0]

def _get_dist(nd, nneighs, mnd_ix, lbs, rks, dss, distance, num_dist=4):
    y_preds = [[] for x in range(num_dist)] # dist, co-rank, min_rank, max_rank

   #- Source node
    nd_ix = np.where(lbs[mnd_ix] == nd)[0][0]
    nd_RKS = rks['rk%i'%(mnd_ix+1)][nd_ix,:]
    nd_DIS = dss['ds%i'%(mnd_ix+1)][nd_ix,:]

    #-Neighbour nodes
    nneighs_ixs = np.where(np.in1d(lbs[1-mnd_ix],nneighs))[0]
    nneighs_RKS = rks['rk%i'%(1-mnd_ix+1)][nneighs_ixs,:].reshape(len(nneighs_ixs),rks['rk%i'%(1-mnd_ix+1)].shape[1])
    nneighs_DIS = dss['ds%i'%(1-mnd_ix+1)][nneighs_ixs,:].reshape(len(nneighs_ixs),dss['ds%i'%(1-mnd_ix+1)].shape[1])
    rk2_rows = lbs[1-mnd_ix][nneighs_ixs]

    for nneigh in nneighs:


        nneigh_ix = np.where(rk2_rows == nneigh)[0][0]
        nneigh_RKS = nneighs_RKS[nneigh_ix]
        nneigh_DIS = nneighs_DIS[nneigh_ix]
        nneigh_target_ix = np.where(lbs[1-mnd_ix] == nneigh)[0][0]
        dist = None

        try:
            r1 = np.where(nd_RKS==nneigh_target_ix)[0][0]
            d1 =  nd_DIS[r1]
            r1 = r1/len(lbs[1-mnd_ix])
        except IndexError:
            r1 = (len(nd_RKS)+1)/len(lbs[1-mnd_ix])
            if distance == 'cosine':
                d1 = 0
            else:
                d1 = max(nd_DIS) + np.std(nd_DIS)

        try:
            r2 = np.where(nneigh_RKS==nd_ix)[0][0]
            d2 = nneigh_DIS[r2]
            r2 = r2/len(lbs[mnd_ix])

        except IndexError:
            r2 = (len(nneigh_RKS)+1)/len(lbs[mnd_ix])
            if distance == 'cosine':
                d2 = 0
            else:
                d2 = max(nneigh_DIS)+ np.std(nneigh_DIS)

        dist = np.mean([d1,d2])

        #--Distance 1
        if distance == 'euclidean':
            y_preds[0].append(1/(1+dist))
        elif distance == 'cosine':
            y_preds[0].append(dist)
        else:
            y_preds[0].append(1-dist)

        #--Distance 2
        y_preds[1].append(1-gmean([r1,r2]))
        #--Distance 3
        y_preds[2].append(1-min([r1,r2]))
        #--Distance 4
        y_preds[3].append(1-max([r1,r2]))
    return y_preds

ns2splits = []
with open(scratch_path+'/n2split.tsv','r') as f:
    for l in f:
        h = l.rstrip().split('\t')
        if h[1] not in nodes_uvs: continue
        nneigh = h[2].split(',')
        no_nneig = [x.split(',') for x in h[3].split('|')]

        ns2splits.append((int(h[0]),h[1],nneigh,no_nneig))

with open(scratch_path+'/lbs.pkl','rb') as f:
    lbs = pickle.load(f)
rks = h5py.File(scratch_path+'/rks.h5','r')
dss = h5py.File(scratch_path+'/dss.h5','r')

with open(scratch_path+'/_cluster/_nd2AUC/%s.tsv'%str(uuid.uuid4()), 'w') as o1, open(scratch_path+'/_cluster/_all_preds/%s.tsv'%str(uuid.uuid4()), 'w') as o2:
    for _, n, nneighs, no_nneigs in tqdm(ns2splits, desc='Nodes'):
        aucs_per_n = []
        worst_aucs_per_n = []
        all_preds = []

        y_preds_pos = _get_dist(n, nneighs, _, lbs, rks, dss, distance)
        y_truth_pos = [1]*len(y_preds_pos[0])
        if type(no_nneigs[0]) != list and type(no_nneigs[0]) != np.ndarray:
            no_nneigs = [no_nneigs]

        auc_boots = [[] for x in range(len(y_preds_pos))]
        all_y_preds_neg,all_y_truths_neg = [],[]
        for neg_boots in no_nneigs:
            y_preds_neg = _get_dist(n, neg_boots, _, lbs, rks, dss, distance)
            y_truth_neg = [0]*len(y_preds_neg[0])
            all_y_preds_neg.append(y_preds_neg)
            all_y_truths_neg.append(y_truth_neg)
            for i in range(len(auc_boots)):
                auc_boots[i].append(roc_auc_score(y_truth_pos+y_truth_neg, y_preds_pos[i]+y_preds_neg[i]))


        for i in range(len(auc_boots)):
            aucs_per_n.append(np.mean(auc_boots[i]))
            worst_aucs_per_n.append(np.min(auc_boots[i]))
            all_preds.append(y_preds_pos[i]+all_y_preds_neg[np.argsort(auc_boots[i])[int(len(auc_boots[i])/2)-1]][i]) #getting med of neg aucs

        truths = y_truth_pos+y_truth_neg

        o1.write('%s\\t%s\\t%s\\t%s\\n'%(str(_), str(n), '\\t'.join(['%.4f'%x for x in aucs_per_n]), '\\t'.join(['%.4f'%x for x in worst_aucs_per_n])))
        for i in range(len(all_preds[0])):
            o2.write('%i\\t'%truths[i]+'\\t'.join(['%.4f'%all_preds[x][i] for x in range(len(all_preds))])+'\\n')

"""

with open(clust_scratch+'/tmp_script.py','w') as o:
    o.write(tmp_script)

#Executing cluster
all_nodes = list(all_nodes)
elements = []
for x in np.arange(0,len(all_nodes),split_nodes):
    elements.append((scratch_path,distance,set(all_nodes[x:x+split_nodes])))

#Getting cluster parameters
cluster = HPC(**cluster_config)
#--Updating cluster params
cluster_params = {}
cluster_params['job_name'] = 'emb_validation'
cluster_params["jobdir"] = clust_scratch
cluster_params['memory'] = 4
cluster_params['cpu'] = 2
cluster_params["wait"] = True
cluster_params["elements"] = elements
cluster_params["num_jobs"] = len(elements)

command = "singularity exec {} python3 {} <TASK_ID> <FILE>".format(
singularity_image,
clust_scratch+'/tmp_script.py')
cluster.submitMultiJob(command, **cluster_params)

#Rewrite  files
out_name = validation_path+'/nd2AUC.tsv'
with open(out_name, 'w') as outfile:
    outfile.write('position\tnode\tdist-avg\tavg_pval-avg\tmin_pval-avg\tmax_pval-avg\tdist-worst\tavg_pval-worst\tmin_pval-worst\tmax_pval-worst\n')
    for filename in [clust_scratch+'/_nd2AUC/'+x for x in os.listdir(clust_scratch+'/_nd2AUC')]:
        with open(filename, 'r') as readfile:
            shutil.copyfileobj(readfile, outfile)
shutil.rmtree(clust_scratch+'/_nd2AUC/')


out_name = scratch_path+'/all_preds.tsv'
with open(out_name, 'w') as outfile:
    outfile.write('y_truth\tdistance\tavg_pval\tmin_pval\tmax_pval\n')
    for filename in [clust_scratch+'/_all_preds/'+x for x in os.listdir(clust_scratch+'/_all_preds')]:
        with open(filename, 'r') as readfile:
            shutil.copyfileobj(readfile, outfile)
        os.remove(filename)
shutil.rmtree(clust_scratch+'/_all_preds/')

#-------------------------------------------

#------ PLOTTING----------

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

#-- Plotting AUROC distribution
m_avg,m_worst = [],[]
with open(validation_path+'/nd2AUC.tsv','r') as f:
    header = f.readline().split('\t')
    header[2:] = [x.split('-')[0] for x in header[2:]]
    for l in f:
        h = l.rstrip().split('\t')
        mnd = mnds[int(h[0])]
        m_avg+=zip([mnd]*len(h[2:6]),header[2:6],map(float,h[2:6]))
        m_worst+=zip([mnd]*len(h[6:]),header[6:],map(float,h[6:]))
m = [m_avg,m_worst]

with sns.axes_style("whitegrid"):
    fig = plt.subplots(figsize=(12,5),dpi=100)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    axs = [ax1,ax2]
    for i in range(len(m)):
        df = pd.DataFrame(m[i],columns=['mnd','Distance','AUROC'])
        palette = dict(zip(np.unique(df['mnd']), col30))
        sns.boxplot(x = 'Distance', y='AUROC', hue='mnd',data=df, ax=axs[i], flierprops={'marker':'o', 'markersize':2}, palette=palette)

    yticks = [0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
    ax1.set_yticks(yticks)

    ax2.set_yticks(yticks)
    ax2.set_yticklabels(['']*len(yticks))
    ax2.set_ylabel('')
    ax1.set_title('AUROC Bootstrap avg.')
    ax2.set_title('AUROC Bootstrap worst')
    ax1.set_ylim(yticks[0],yticks[-1]+.005)
    ax2.set_ylim(yticks[0],yticks[-1]+.005)
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig(validation_path+'/AUROC_per_nd.png', bbox_inches='tight')

#Global AUROC
all_aurocs = []
df = pd.read_csv(scratch_path+'/all_preds.tsv',sep='\t')
models = df.columns[1:]
plt.figure(figsize=(4,4),dpi=100)
for ix, model in enumerate(models):
    fpr,tpr,thr = roc_curve(df['y_truth'],df[model])
    auroc = roc_auc_score(df['y_truth'],df[model])
    plt.plot(fpr,tpr,label='%s (%.2f)'%(model,auroc), color=col30[ix])
    all_aurocs.append((model,auroc))

cnt = Counter(df['y_truth'])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Global ROC (0:%i | 1:%s)'%(cnt[0],cnt[1]))
plt.xlim(-0.01,1.01)
plt.ylim(-0.01,1.01)
plt.plot([0,1],[0,1],linestyle='--',color='navy')
plt.legend(loc=4, handlelength=0.8)
plt.savefig(validation_path+'/global_ROC.png',bbox_inches='tight')

with open(validation_path+'/global_aurocs.txt','w') as o:
    for r in all_aurocs:
         o.write('%s\t%.3f\n'%(r[0],r[1]))
