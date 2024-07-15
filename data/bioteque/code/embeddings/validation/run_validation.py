root_path = '../../../'
code_path = root_path+'code/embeddings/'
emb_path = root_path+'embeddings/'


"""
It runs the following validations:

    1) tSNE projections
    2) faiss nearest neighbors calculation
    3) distance validation (euclidean and cosine)
    4) cross-intra validation
    5) cross-inter validation
    6) cross-mnd validation
    7) embedding card

"""
import sys
import os
import subprocess
import shutil
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
sys.path.insert(1, code_path)

from utils.embedding_utils import read_embedding, get_embedding_universe, RNDuplicates
from utils.graph_preprocessing import read_edges
from validation.val_utils import tsne_val

nneigh_script = code_path+'/distances/get_nneigh.py'
roc_val_script = code_path+'/validation/roc_validation.py'
pval_script = code_path+'/validation/pval_validation.py'
cross_intra_script = code_path+'/validation/cross_characterization/intra_charact.py'
cross_inter_script = code_path+'/validation/cross_characterization/inter_charact.py'
cross_mnd_script = code_path+'/validation/cross_characterization/mnd_charact.py'
emb_card_script = code_path+'/validation/run_emb_card.py'

# ------------------ Plot options ---------------------
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
#-----------------------------

current_path = sys.argv[1]
emb_path = '/'+'/'.join([x for x in current_path.split('/') if x!=''][:-3]) +'/'
window = 5
distances = ['euclidean','cosine']

n_bootstrap = 10
_metapath,_dataset = [x.strip() for x in current_path.split('/') if x != ''][-2:]
sys.stderr.write('Running validation for: %s || %s...\n'%(_metapath, _dataset))

#--Making directories
if not os.path.exists(current_path+'/validation'):
    os.mkdir(current_path+'/validation')
validation_path = current_path+'/validation/w%s/'%str(window)
if not os.path.exists(validation_path):
    os.mkdir(validation_path)

#######################
# (1) tSNE projection
#######################

sys.stderr.write('1) Plotting tSNE...\n')
tsne_path = validation_path+'/tSNE/'

if not os.path.exists(tsne_path):
    os.mkdir(tsne_path)
if not os.path.exists(tsne_path+'/default'):
    os.mkdir(tsne_path+'/default')
if not os.path.exists(tsne_path+'/degree'):
    os.mkdir(tsne_path+'/degree')

mnds = sorted([x[:-4] for x in os.listdir(current_path+'/nodes')])
mnd2col = dict(zip(mnds, ['#5078dc','#fa6450']))
k2col = {}

#--Getting node2degree
with h5py.File(current_path+'/nd2st.h5','r') as f:
    id2nd = dict(zip(f['id'][:], f['nd'][:].astype(str)))

nd2dg = {}
for h in read_edges(current_path+'/edges.h5'):
    n1,n2 = id2nd[h[0]], id2nd[h[1]]
    if n1 not in nd2dg:
        nd2dg[n1] = 0
    if n2 not in nd2dg:
        nd2dg[n2] = 0
    nd2dg[n1]+=1
    nd2dg[n2]+=1
del id2nd

#--Mnd specific
for mnd in mnds:
    _e = read_embedding(current_path+'/embs/w%s/%s.h5'%(window,mnd), just_the_matrix=True)
    keys = np.array(get_embedding_universe(current_path,mnd=mnd))

    #----Removing near duplicates
    sys.stderr.write('--> Removing near duplicates...\n')
    rnn = RNDuplicates()
    ixs = rnn.remove(_e, just_keys=True)
    _e = dict(zip(keys[ixs],_e[ixs]))

    tsne_val(_e,
             opath1=tsne_path+'/default/%s_tsne.png'%mnd,
             opath2=tsne_path+'/degree/%s_tsne.png'%mnd,
             nd2degree = nd2dg,
             title='%s'%mnd,
             )
    if len(mnds) > 1:
        k2col.update({x:mnd2col[mnd] for x in _e})

#--ALL metanodes
if len(mnds) > 1:
    _e = {}

    for mnd in mnds:
        _e.update(dict([(x,y) for x,y in read_embedding(current_path+'/embs/w%s/%s.h5'%(window,mnd)).items() if x in k2col])) #- I use the keys that are in k2col as they have no near duplicates

    legend_handles = [Line2D([0], [0], marker='o', color=color,label='%s'%''.join([_ for _ in mnd.split('__') if _ != '']), markerfacecolor=color, markersize=7, linewidth=0)
                   for mnd,color in mnd2col.items()] #tags borders (__) are removed as python don't allow them to be on the legend

    tsne_val(_e,
             opath1=tsne_path+'/default/tsne.png',
             opath2=tsne_path+'/degree/tsne.png',
             nd2degree = nd2dg,
             nd2col = k2col,
             title='%s | %s'%(mnds[0], mnds[1]),
             legend_handles = legend_handles,
             alpha = 0.5
             )

del nd2dg

################
# (2) FAISS NN
################
sys.stderr.write('2) Getting nearest neighbours\n')
if not os.path.exists(current_path+'/nneigh'):
    subprocess.Popen('python3 %s %s'%(nneigh_script, current_path), shell = True).wait()

###################
# (3) Distance val
###################
if not os.path.exists(validation_path+'/distances'):
    os.mkdir(validation_path+'/distances')

#--Geting max_rk based on size of the network
with h5py.File(current_path+'/nd2st.h5','r') as f:
    _l = f['nd'].shape[0]

if _l < 25000:
    max_rk = 0.75
elif _l < 50000:
    max_rk = 0.5
elif _l < 75000:
    max_rk = 0.25
elif _l < 100000:
    max_rk = 0.1
else:
    max_rk = 5000 / _l #MAX == 5000

for distance in tqdm(distances,leave=False,desc='dist'):
    #3.1) Getting roc validation
    sys.stderr.write('3.1) Running ROC validation (%s)\n'%distance)
    subprocess.Popen('python3 %s -path %s -d %s -w %i -pp %i -r %f -b %i'%(roc_val_script,
                                                                     current_path,
                                                                     distance,
                                                                     window,
                                                                     0, # m++ will not assessed by default,
                                                                     max_rk,
                                                                     n_bootstrap
                                                        ), shell = True).wait()

    #3.2) Getting p-value validation
    sys.stderr.write('3.2) Running p-value validation (%s)\n'%distance)
    subprocess.Popen('python3 %s -path %s -d %s -w %i -pp %i -r %f'%(pval_script,
                                                         current_path,
                                                         distance,
                                                         window,
                                                         0, # m++ will not assessed by default
                                                         max_rk,
                                                        ), shell = True).wait()

    #--Removing validation files (only keeping plots, global AUROCs and flawed_nodes file)
    os.remove(current_path+'/validation/w5/distances/%s/nd2AUC.tsv'%distance)
    os.remove(current_path+'/validation/w5/distances/%s/pval_quartiles.tsv'%distance)
    os.remove(current_path+'/validation/w5/distances/%s/pval_errors.tsv'%distance)
    shutil.rmtree(current_path+'/scratch/validation/')

#####################
# (4) Cross-Intra val
#####################
sys.stderr.write('4) Cross-Intra validation\n')
subprocess.Popen('python3 %s %s %s %s'%(cross_intra_script, _metapath, _dataset, emb_path), shell = True).wait()

#####################
# (5) Cross-Inter val
#####################
sys.stderr.write('5) Cross-Inter validation\n')
subprocess.Popen('python3 %s %s %s %s'%(cross_inter_script, _metapath, _dataset, emb_path), shell = True).wait()

#####################
# (6) Cross-mnd val
#####################
sys.stderr.write('6) Cross-mnd validation\n')
subprocess.Popen('python3 %s %s %s %s'%(cross_mnd_script, _metapath, _dataset, emb_path), shell = True).wait()

#####################
# (7) Embedding card
#####################
sys.stderr.write('7) Embedding card\n')
subprocess.Popen('python3 %s %s'%(emb_card_script, current_path), shell = True).wait()
