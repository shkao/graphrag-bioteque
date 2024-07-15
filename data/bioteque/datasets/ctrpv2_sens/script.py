import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import psycopg2
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps
from transform_data import  merge_duplicated_cells, drug_sens_stratified_waterfall, quantile_norm

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

##------------------------------------------------------------------------------------

# !! Important --> Configure this function according to your host, user and password
def psql_connect(h='your_host',u='your_user',p='your_password'):
    datab = psycopg2.connect(host=h, user=u, password=p, database='pharmacodb')
    return datab

con = psql_connect()
cur = con.cursor()

##------------------------------------------------------------------------------------

#id2dt
dt_of_interest = 'CTRPv2'
id2dt = {}
cur.execute("SELECT * from datasets")
for i,dt in cur:
    if dt == dt_of_interest:
        id2dt[i] = dt

#ci2n
cur.execute("SELECT * from source_cell_names where source_id IN (%s)"%','.join([str(x) for x in list(id2dt)]))
cid2n = {}
for _,i, s, n in cur:
    cid2n[i] = n

#d2ikey
d2ikey = mps.get_pharmacodb2ikey()

#Retriving experimental data
cur = con.cursor()
cur.execute("SELECT * FROM experiments")
exps = {}
for r in cur:
    if r[3] not in id2dt: continue
    if r[1] not in cid2n: continue
    if r[2] not in d2ikey: continue
    r = list(r)
    r[3] = id2dt[r[3]]
    r[2] = d2ikey[r[2]]
    r[1] = cid2n[r[1]]

    exps[r[0]] = [r[1], r[2], r[3]]

#Retriving profiles
cur = con.cursor()
cur.execute("SELECT * FROM profiles t1 where t1.experiment_id IN (%s)"%','.join([str(x) for x in exps]))
m = []
for r in cur:
    m.append(exps[r[0]]+[r[4]])

df = pd.DataFrame(m, columns=['cl','dg','dt','AAC'])

#Creating final matrix
all_cls = np.unique(df['cl'])
all_dgs = np.unique(df['dg'])
ixs = []
m = []
for dg,d in df.groupby('dg'):
    v = np.asarray([np.nan]*len(all_cls))
    if len(set(d['dt'])) > 1:
        d = d[d['dt']==dt_of_interest]

    #Getting mean of replicates
    d = d[['cl','AAC']].groupby('cl').mean().sort_index()
    v[np.where(np.in1d(all_cls,d.index.values))[0]] = list(d['AAC'])
    m.append(v)
    ixs.append(dg)

df_sns = 1-pd.DataFrame(m, index=ixs, columns=all_cls).T

#Mapping CLs
cl2ID = dict(zip(df_sns.index.values, mps.cl2ID(df_sns.index.values)))
df_sns.index = [cl2ID[cl] for cl in df_sns.index.values]
df_sns = df_sns.drop(index=[None]) #removing unmapped clls

# Merge Duplicate Samples By Rows (by taking the mean)
df_sns = merge_duplicated_cells(df_sns, 'row', 'mean')

# Merge Duplicate Drugs By Columns
df_sns = merge_duplicated_cells(df_sns, 'column', 'mean')

#Binarizing
df_sns = df_sns.transform(drug_sens_stratified_waterfall, axis = 0, ternary=False,
                          min_cls=0.01, max_cls=0.2, apply_uncertainty_at=0.8, min_sens_cutoff=0.9)

#Getting sns cpds per cell line
dgs = np.asarray(df_sns.columns)
sns = set([])
for cl , sens in df_sns.iterrows():
    s = dgs[sens ==1]
    sns.update(zip([cl]*len(s),s))

# Writing output
with open(out_path+'/CLL-sns-CPD/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\n')
    for r in sorted(sns):
        o.write('%s\t%s\n'%(r[0],r[1]))

sys.stderr.write('Done!\n')
