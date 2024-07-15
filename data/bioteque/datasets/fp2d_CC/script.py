import sys
import os
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps
from tqdm import tqdm
#                      --------------------------------------------------------------
#---------------------- To run this you need access to the Chemical Checker database --------------------------------------------------------
#                      --------------------------------------------------------------

# #-------------Parameters--------------
#
# jc_cutoff = 0.7
#
# #-------------------------------------
#
# def jaccard(set1,set2):
#     """Given 2 sets, calculates the Jaccard index"""
#
#     if type(set1) != set:
#         set1 = set(set1)
#     elif type(set2) != set:
#         set2 = set(set2)
#     return len(set1 & set2)/len(set1 | set2)
#
# def psql_connect(db, h='your_host',u='your_user',p='your_password'):
#     datab = psycopg2.connect(host=h, user=u, password=p, database=db)
#     return datab
#
# #Retriving all drugs
# all_drugs = set([])
# with open('../../../graph/nodes/Compound.tsv','r') as f:
#     next(f)
#     all_drugs = sorted(set([l.rstrip().split('\t')[0] for l in f]))
#
# #Connecting to mosaic db (--> Chemical checker raw data)
# con = psql_connect('mosaic')
# db = conn.cursor()
#
# #Retriving physchem attributes
# sys.stderr.write('Retriving fingerprint 2D signatures from the database...')
# db.execute("SELECT * FROM fp2d WHERE inchikey = ANY('{%s}'::text[])"%','.join(all_drugs))
# sys.stderr.write('Done!')
# dg2fp = {}
# for h in db.fetchall():
#     dg = h[0]
#     fp = h[1].split(',')
#     dg2fp[dg] = fp
#
# conn.close()
#
# with open('./edges.tsv','w') as o:
#     o.write('n1\tn2\ttanimoto\n')
#     for i in tqdm(range(len(all_drugs) - 1),desc='Writing'):
#         dg1 = all_drugs[i]
#         if dg1 not in dg2fp: continue
#         s1 = set(dg2fp[dg1])
#
#         for j in range(i+1, len(all_drugs)):
#             dg2 = all_drugs[j]
#             if dg2 not in dg2fp: continue
#             s2 = set(dg2fp[dg2])
#             jc = jaccard(s1,s2)
#             if jc >= jc_cutoff:
#                 o.write('%s\t%s\t%.3f\n'%(dg1,dg2,jc))

#--------------------------------------------------------------------------------

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]
opath = out_path+'/CPD-sim-CPD/%s.tsv'%source


if os.path.exists(opath):
    if os.path.islink(opath):
        os.unlink(opath)
    else:
        os.remove(opath)

os.symlink(os.path.abspath('./edges.tsv'),os.path.abspath(opath))
sys.stderr.write('Done!\n')
