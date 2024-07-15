root_path = '../../../'
code_path = root_path+'code/embeddings/'
emb_path = root_path+'embeddings/'
import os
import sys
import h5py

sys.path.insert(1, code_path)
from distances.embedding_distances import create_faiss_index

p  = sys.argv[1]

possible_datasets = ['m','m++']

#--making directories
if not os.path.exists(p+'/nneigh'):
    os.mkdir(p+'/nneigh')

folders = os.listdir(p+'/embs')

for folder in folders:

    files = os.listdir(p+'/embs/'+folder)
    out_path = p+'/nneigh/%s/'%folder

    for file in files:
        mnd = file[:-3]

        with h5py.File(p+'/embs/%s/%s'%(folder,file),'r') as f:
            for name in possible_datasets:
                if name not in f.keys(): continue
                m = f[name][:]

                if name == 'm++':
                    out_path = p+'/nneigh/++%s/'%folder
                else:
                    out_path = p+'/nneigh/%s/'%folder

                if not os.path.exists(out_path):
                    os.mkdir(out_path)

                #--Euclidean
                create_faiss_index(m, distance='euclidean', write_to = out_path+'/%s.l2.faiss'%mnd)
                #--Cosine
                create_faiss_index(m, distance='cosine', write_to = out_path+'/%s.cos.faiss'%mnd)
