root_path = '../../../../'
code_path = root_path+'code/embeddings/'
emb_path = root_path+'embeddings/'
recapitulation_path = root_path+'validation/cross_characterization/recapitulation_pairs/metaedges/'

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
from utils.graph_preprocessing import get_neg_edges_from_egs_simple

def get_emb(mpath, dt, mnds, emb_path):
    emb = {}
    for mnd in set(mnds):
        emb.update(read_embedding(mpath=mpath, dt=dt, mnd=mnd, w=5, pp=False, emb_path=emb_path))
    return emb

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

#------------------
if  __name__ == "__main__":

    #Verifying argments
    mpath = sys.argv[1]
    dt = sys.argv[2]
    if len(sys.argv) >3:
        emb_path = sys.argv[3]
    else:
        emb_path = emb_path

    #Checking if there are tags. If so, skipping this validation as probably does not make sense
    _args = parse_parameter_file(emb_path+'/%s/%s/%s/parameters.txt'%(mpath[:3],mpath,dt))
    if _args['tag_source']:
        sys.stderr.write('A source tag "%s" was detected. Exiting...\n '%_args['tag_source'])
    elif  _args['tag_target']:
        sys.stderr.write('A source tag "%s" was detected. Exiting...\n '%_args['tag_target'])
    else:
        mnd1 = _args['source']
        mnd2 = _args['target']
        ref_nds = sorted([mnd1, mnd2])
        undirected_metaedges = get_undirected_megs()

        if not os.path.exists(emb_path+'/%s/%s'%(mpath[:3], mpath)):
            sys.exit('The metapath "%s" does not exist! Exiting...\n'%mpath)
        if not os.path.exists(emb_path+'/%s/%s/%s'%(mpath[:3], mpath, dt)):
            sys.exit('The dataset "%s" does not exist! Exiting...\n'%dt)

        #Making directories
        val_path = emb_path+'/%s/%s/%s/validation/w5/cross_charact/'%(mpath[:3], mpath, dt)
        if not os.path.exists(val_path):
            os.mkdir(val_path)

        inter_path = val_path + '/inter/'
        if not os.path.exists(inter_path):
            os.mkdir(inter_path)

        #Getting embeddings
        emb = get_emb(mpath, dt, ref_nds, emb_path=emb_path)

        #Iterating for each metaedge
        R = []
        megs_list = [me for me in os.listdir(recapitulation_path) if sorted([me[:3],me[-3:]]) == ref_nds]
        for me in tqdm(megs_list, desc='metaedges'):

            #--Metaedge variables
            is_undirected = True if me in undirected_metaedges else False
            dts = [dt[:-4] for dt in os.listdir(recapitulation_path+me)]

            #--Making directories
            opath = inter_path+'/%s/'%me
            if not os.path.exists(opath):
                os.mkdir(opath)
            #if os.path.exists(opath+'/roc_validation.png'): continue

            #--Plot options
            if len(dts) == 1:
                gridspec_kw = {"width_ratios" : [9.5, 0.5]}
                fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(4.21, 4), gridspec_kw=gridspec_kw)
            else:
                gridspec_kw = {"width_ratios" : [9, 1]}
                fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(4.44, 4), gridspec_kw=gridspec_kw)
            plt.subplots_adjust(wspace=0.0, hspace=0)


            cov_uvs = []
            for ix, dt in tqdm(enumerate(sorted(dts)), desc=me, leave=False):

                #Getting edges
                edges = set([])
                u = set([])
                u_val = set([])
                n_edges_val = 0
                covered_edges = 0

                with open(recapitulation_path+'/%s/%s.tsv'%(me,dt), 'r') as f:
                    for l in f:
                        if l.startswith('n1'): continue
                        h = l.rstrip().split('\t')
                        u_val.update([h[0], h[1]])
                        n_edges_val+=1

                        if h[0] in emb and h[1] in emb:
                            edges.add((h[0], h[1]))
                            u.update([h[0], h[1]]) #The edge files should not have repetitions nor inversions...
                            covered_edges+=1

                #Getting coverages
                n_nodes_val = len(u_val)
                cov_nodes_val = len(u & u_val)/len(u_val)
                cov_edges_val = covered_edges/n_edges_val
                cov_uv  = len(u & set(emb))/len(emb)
                del u, u_val
                if cov_uv == 0: continue
                cov_uvs.append(cov_uv)
                no_coverage_flag = False

                #--Getting neg edges
                neg_edges = get_neg_edges_from_egs_simple(edges, is_undirected=is_undirected)

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
                        dt_name = rename_dt[dt] if dt in rename_dt else dt
                        dt_name = dt_name[:11].rstrip('_') if len(dt_name) > 11 else dt_name
                        label = '%.2f|%.2f %s...'%(aurocs[0], aurocs[1], dt_name)

                    else:
                        label = None

                    #--Plotting
                    axes[0].plot(fpr, tpr, color=col30[ix], linestyle=ls, lw=2,label = label)

                if no_coverage_flag: continue

                #Keeping results
                R.append([me, dt, '%.3f'%(aurocs[0]), '%.3f'%(aurocs[1]), '%.6f'%cov_uv, '%.6f'%cov_nodes_val, '%i'%n_nodes_val, '%.6f'%cov_edges_val,'%i'%n_edges_val])

            if len(cov_uvs) == 0 or no_coverage_flag is True:
                plt.close()
                continue

            #Ploting coverages
            if len(dts) == 1:
                width = 0.3
            else:
                width = 0.15
            for ix,cov_uv in enumerate(cov_uvs):
                axes[1].bar([0.25*ix], [cov_uv], width=width, color=col30[ix], edgecolor='black')

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

            axes[0].set_title(me,y=1.02)

            #--Coverages plot options
            axes[1].grid(linestyle="-.", color='black', lw=0.3, axis='y')
            if len(dts) == 1:
                axes[1].set_xlim(-0.15,0.15)
            else:
                axes[1].set_xlim(-0.15,(0.25*(len(dts)-1)+0.15))
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
            R.to_csv(inter_path+'/results.csv')
