root_path = '../../../../'
code_path = root_path+'code/embeddings/'

recapitulation_path = root_path+'validation/cross_characterization/recapitulation_pairs/metanodes/'
onto_path = root_path+'/metadata/ontologies/'

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
from utils.embedding_utils import read_embedding
from validation.val_utils import  col30
from utils.graph_preprocessing import get_neg_edges_from_egs_simple
from utils.utils import node2abbr, parse_parameter_file
code_path2 = root_path+'code/kgraph/'
sys.path.insert(1, code_path2)
from utils import ontology_processing as DSP
#----------------------------
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
#--------------------------------
def get_inheritable_ontology_dict(node_type, onto_path=onto_path):

    if len(node_type) == 3:
        node_type = node2abbr(node_type, reverse=True)
    node_type = node_type.lower().capitalize()

    if node_type not in os.listdir(onto_path):
        return None

    else:
        child2parent = []
        universe = set([])

        if node_type == 'Disease':
            file = 'doid.tsv'
        elif node_type == 'Pathway':
            file = 'reactome.tsv'
        else:
            file = os.listdir(onto_path+'/%s'%node_type)[0]

        with open(onto_path+'/%s/%s'%(node_type,file),'r') as f:
            f.readline()
            for l in f:
                h = l.rstrip('\n').split('\t')
                child2parent.append(h)
                universe.update(h)

        return {'c2p':child2parent,
                'universe':universe}

def depropagate(M, onto_dict):

    if type(M[0]) != tuple:
        M = list(map(tuple, M))

    #Depropagating
    m = set([x for x in M if x[0] in onto_dict['universe'] and x[1] in onto_dict['universe']])
    final_associations = set(M) & DSP.depropagate(m,onto_dict['c2p'],onto_dict['c2p'])

    #Re-adding those edges that were not mapped to the ontology
    final_associations.update(set(M) - m)

    return final_associations

#------------------
if  __name__ == "__main__":

    #Verifying argments
    mpath = sys.argv[1]
    dt = sys.argv[2]
    if len(sys.argv) >3:
        emb_path = sys.argv[3]
    else:
        emb_path = emb_path

    _args = parse_parameter_file(emb_path+'/%s/%s/%s/parameters.txt'%(mpath[:3],mpath,dt))
    mnd1 = _args['source'] if _args['tag_source'] is None else _args['tag_source'] + _args['source']
    mnd2 = _args['target'] if _args['tag_target'] is None else _args['tag_target'] + _args['target']
    ref_mnds = sorted([mnd1, mnd2])

    if not os.path.exists(emb_path+'/%s/%s'%(mpath[:3], mpath)):
        sys.exit('The metapath "%s" does not exist! Exiting...\n'%mpath)
    if not os.path.exists(emb_path+'/%s/%s/%s'%(mpath[:3], mpath, dt)):
        sys.exit('The dataset "%s" does not exist! Exiting...\n'%dt)

    #Making directories
    val_path = emb_path+'/%s/%s/%s/validation/w5/cross_charact/'%(mpath[:3], mpath, dt)
    if not os.path.exists(val_path):
        os.mkdir(val_path)

    mnds_path = val_path + '/mnd/'
    if not os.path.exists(mnds_path):
        os.mkdir(mnds_path)

    #Getting possible embeddings
    epaths = emb_path+'/%s/%s/%s/embs/'%(mpath[:3], mpath, dt)

    #--Only w5 without m++ is considered!
    epaths = [epaths+x for x in os.listdir(epaths) if 'w5' in x]
    dataset_names = ['m']

    for mnd in ref_mnds:
        vals = os.listdir(recapitulation_path+'/%s'%mnd.split('__')[-1])
        for epath in epaths:
            for name in dataset_names:

                #Getting embeddings
                try:
                    emb = read_embedding(final_path=epath+'/%s.h5'%mnd, name=name, emb_path=emb_path)
                    emb = {e.split('__')[-1]:emb[e] for e in emb} #Removing tags to allow the matching with the validation files

                except KeyError:
                    continue

                if name == 'm++':
                    mnd_path = mnds_path+'/++%s'%mnd
                else:
                    mnd_path = mnds_path+'/%s'%mnd

                if not os.path.exists(mnd_path):
                    os.mkdir(mnd_path)

                #Getting Ontology dict to depropagate (if needed)
                onto_dict = get_inheritable_ontology_dict(mnd.split('__')[-1])

                R = []
                for me in vals:

                    opath = mnd_path+'/%s'%me
                    if not os.path.isdir(opath):
                        os.mkdir(opath)
                    #if os.path.exists(opath+'/roc_validation.png'): continue

                    val_folder = recapitulation_path+'/%s/%s/'%(mnd.split('__')[-1], me)
                    dts = os.listdir(val_folder)

                    #--Plot options
                    if len(dts) == 1:
                        gridspec_kw = {"width_ratios" : [9.5, 0.5]}
                        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(4.21, 4), gridspec_kw=gridspec_kw)
                    else:
                        gridspec_kw = {"width_ratios" : [9, 1]}
                        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(4.44, 4), gridspec_kw=gridspec_kw)

                    plt.subplots_adjust(wspace=0.0, hspace=0)

                    #Getting edges
                    cov_uvs = []
                    for ix, val_dt in tqdm(enumerate(sorted(dts)), desc=me, leave=False):
                        val_name = val_dt[:-4]

                        u = set([])
                        u_val = set([])
                        edges = []

                        n_edges_val = 0
                        covered_edges = 0
                        MAX_EDGES = 1e6
                        with open(val_folder+val_dt, 'r') as f:
                            for l in tqdm(f, desc='Reading validation', leave=False):

                                h = l.rstrip().split('\t')
                                if h[0].startswith('n1'): continue
                                if h[2] != '1': continue #Only keeping positive as negatives are recalculated on the fly

                                u_val.update([h[0], h[1]])
                                n_edges_val+=1

                                if h[0] in emb and h[1] in emb:
                                    u.update([h[0], h[1]])
                                    covered_edges+=1
                                    edges.append([h[0], h[1]])
                                if covered_edges >= MAX_EDGES: break #If I already have 1M edges to valide I stop getting more to prevent memory problems

                        #Depropagate (if needed)
                        if onto_dict is not None and len(edges) > 0:
                            edges = set(depropagate(list(edges),onto_dict=onto_dict))

                        #Getting neg edges
                        neg_edges = get_neg_edges_from_egs_simple(edges, is_undirected=True, disable_tqdm=True)

                        #Getting coverages
                        n_nodes_val = len(u_val)
                        cov_nodes_val = len(u & u_val)/len(u_val)
                        cov_edges_val = covered_edges/n_edges_val
                        cov_uv = len(u & set(emb))/len(emb)
                        del u, u_val
                        if cov_uv == 0: continue
                        cov_uvs.append(cov_uv)
                        no_coverage_flag = False

                        #--Getting dist
                        aurocs = []
                        for dist_fn, ls in [(cosine, ':'), (euclidean, '-')]:
                            p_dist  = [-dist_fn(emb[e[0]], emb[e[1]]) for e in edges]
                            n_dist = [-dist_fn(emb[e[0]], emb[e[1]]) for e in neg_edges]

                            if len(p_dist) == 0 or len(n_dist) == 0:
                                no_coverage_flag = True
                                break

                            #Calculating ROC curve
                            fpr,tpr,thr = roc_curve([1]*len(p_dist) + [0]*len(n_dist), p_dist+n_dist)
                            aurocs.append(auc(fpr,tpr))

                            if dist_fn == euclidean:
                                rename_dt = {
                                'depmap_agreement_ccr': 'depmap_ccr',
                                'depmap_agreement_ceres': 'depmap_ceres',
                                'disgenet_curated': 'cur_disgenet',
                                'disgenet_inferred': 'inf_disgenet',
                                'drugbank_pd':'pd_drugbank',
                                'drugbank_pk':'pk_drugbank'}

                                dt_name = rename_dt[val_name] if val_name in rename_dt else val_name
                                dt_name = dt_name[:11].rstrip('_') if len(dt_name) > 11 else dt_name
                                label = '%.2f|%.2f %s...'%(aurocs[0], aurocs[1], dt_name)

                            else:
                                label = None

                            #--Plotting
                            axes[0].plot(fpr, tpr, color=col30[ix], linestyle=ls, lw=2,label = label)

                        if no_coverage_flag: continue

                        #Keeping results
                        R.append([me, val_name, '%.3f'%(aurocs[0]), '%.3f'%(aurocs[1]), '%.6f'%cov_uv, '%.6f'%cov_nodes_val, '%i'%n_nodes_val, '%.6f'%cov_edges_val,'%i'%n_edges_val])

                    if len(cov_uvs) == 0 or no_coverage_flag is True:
                        plt.close()
                        continue

                    #Plotting coverages
                    if len(dts) == 1:
                        width = 0.3
                    else:
                        width = 0.15
                    for ix,cov_uv in enumerate(cov_uvs):
                        axes[1].bar([0.25*ix], [cov_uv], width=width, color=col30[ix], edgecolor='black')

                    #Final plotting options

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
                    axes[0].set_title(me,y=1.00)

                    #--Coverages plot options
                    axes[1].grid(linestyle="-.", color='black', lw=0.3, axis='y')
                    #axes[1].set_xlim(min(x)-0.25,max(x2)+0.25)
                    if len(dts) == 1:
                        axes[1].set_xlim(-0.15,0.15)
                    else:
                        axes[1].set_xlim(-0.15,(0.25*(len(dts)-1)+0.15))
                    axes[1].set_ylim(-0.003, 1.003)
                    #axes[1].set_title('Cov.')
                    axes[1].set_ylabel('')
                    axes[1].set_xlabel('')
                    axes[1].set_xticks([])
                    #axes[1].set_xticks([np.median(x), np.median(x2)])
                    #axes[1].set_xticklabels(['E', 'P/N'])
                    axes[1].set_yticks(np.arange(0,1.1,0.1))
                    axes[1].set_yticklabels([])
                    axes[1].yaxis.tick_right()
                    axes[1].tick_params(axis='y', which='both',right=False)

                    #--Legend plot options
                    if len(R) > 0:
                        aurocs = [float(j) for i in R[-len(R):] for j in i[2].split('|')]
                        if max(aurocs) < 0.7 and min(aurocs) < 0.4:
                            labels, handles = zip(*sorted(zip(*axes[0].get_legend_handles_labels()),
                                                          key=lambda t: max([float(i) for i in t[1].split()[0].strip().split('|')]), reverse=True))
                            lg =axes[0].legend(loc=2,frameon=False, bbox_to_anchor=(-0.03,1.03), handlelength=0.8)
                        else:
                            labels, handles = zip(*sorted(zip(*axes[0].get_legend_handles_labels()),
                                                          key=lambda t: max([float(i) for i in t[1].split()[0].strip().split('|')]), reverse=True))
                            axes[0].legend(labels, handles, loc=4, frameon=False, bbox_to_anchor=(1.04,-0.05), handlelength=0.8)

                    plt.savefig(opath+'/roc_validation.png', bbox_inches='tight')
                    plt.close()

                #Writing results
                if len(R) > 0:
                    R = sorted(R, key = lambda x: max([float(i) for i in x[2].split('|')]), reverse=True)
                    R = pd.DataFrame(R, columns = ['','dt', 'cos.auroc','euc.auroc', 'nodes_cov', 'val_nodes_cov', 'val_nodes', 'val_edges_cov', 'val_edges'])
                    R = R.set_index('')
                    R.to_csv(mnd_path+'/results.csv')
