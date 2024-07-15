import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import psycopg2
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#-------------Parameters--------------

datasets_of_interest = set(['GDSC1000','CCLE','CTRPv2','gCSI'])
min_pvalue = 0.01
max_gns = 250

#-------------------------------------

# !! Important --> Configure this function according to your host, user and password
def psql_connect(h='your_host',u='your_user',p='your_password'):
    datab = psycopg2.connect(host=h, user=u, password=p, database='pharmacodb')
    return datab
    
def qstring(query):
    con = psql_connect()
    con.set_isolation_level(0)
    cur = con.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    con.close()
    return rows
    
def fetch_drug_gene_correlations(min_pvalue , dts):
    cmd = '''
          SELECT t1.drug_id, t2.gene_name, t1.estimate, t1.pvalue,  t1.dataset_id
            FROM gene_drugs t1, genes t2
              WHERE t1.gene_id = t2.gene_id
                AND t1.pvalue < %.5f
                AND t1."mDataType" = 'mRNA'
                AND t1.dataset_id IN (%s)
          ''' %(min_pvalue, ','.join(map(str,dts)))
    R = qstring(cmd)
    return R

def get_final_statistic(data):

    d = np.asarray(data)
    #If gCSI, I get it (I trust more in this database)
    if dt2id['gCSI'] in set(d[:,-1]):
        return d[np.where(d[:,-1]==dt2id['gCSI'])[0][0]][2]
    else:
        #Checking the direction in each database
        p = len(np.where(d[:,2]>0)[0])
        n = len(np.where(d[:,2]<0)[0])

        #If there are the same number of positive than negative evidence I take the average of the correlation
        if p == n:
            return np.mean(d[:,2])

        #If there are more evidence of one direction I take the result with the lower p-value
        if p > n:
            d = d[d[:,2] > 0]

        elif n > p:
            d = d[d[:,2] < 0]

        return  d[np.argmin(d[:,3]),2]

#Accessing data
con = psql_connect()
cur = con.cursor()

#--Getting mappings
g2upg = mps.get_gene2updatedgene()
g2up = mps.get_gene2unip()
d2ikey = mps.get_pharmacodb2ikey()

#Datasets
dt2id = {}
cur.execute("SELECT * from datasets")
for i,dt in cur:
    if dt in datasets_of_interest:
        dt2id[dt] = i

#Fetching drug_gene_correlations
m = []
sys.stderr.write('Fetching drug-gene correlations...\n')
for r in fetch_drug_gene_correlations(min_pvalue = min_pvalue, dts=list(dt2id.values())):
    m.append(r)
df = pd.DataFrame(m, columns=['did','g','t','p','dt'])

#Combining duplicated drug,gene correlations
m = []
for ix, data in tqdm(df.groupby(['did','g']), desc='Combining databases'):
    if len(data) == 1:
        m.append(list(data.values[0][:3]))
    else:
        m.append(list(ix)+[get_final_statistic(data)])
df = pd.DataFrame(m, columns=['did','g','t'])

#Getting significant associations
gen_ups_cpd = set([])
gen_dws_cpd = set([])
sys.stderr.write('Getting significant associations...\n')
for dg,data in tqdm(df.groupby('did')):

    #Mapping drug
    if dg not in d2ikey: continue
    dg = d2ikey[dg]

    #Sorting data values
    data = data.sort_values('t')

    #--neg
    neg = data.iloc[:max_gns]
    neg = neg[['g','t']][neg['t']<0]
    for g,t in neg.values:

        if g in g2upg:
            g = g2upg[g]
        if g in g2up:
            for up in g2up[g]:
                gen_dws_cpd.add((up,dg, t))

    #--pos
    pos = data.iloc[-max_gns:]
    pos = pos[['g','t']][pos['t']>0]
    for g,t in pos.values:

        if g in g2upg:
            g = g2upg[g]
        if g in g2up:
            for up in g2up[g]:
                gen_ups_cpd.add((up,dg,t))

    assert len(set(neg['g']) & set(pos['g'])) == 0

incongruencies = set([tuple(x[:2]) for x in gen_ups_cpd]) & set([tuple(x[:2]) for x in gen_dws_cpd])
gen_ups_cpd = set([x for x in gen_ups_cpd if tuple(x[:2]) not in incongruencies])
gen_dws_cpd = set([x for x in gen_dws_cpd if tuple(x[:2]) not in incongruencies])

#Writing
with open(out_path+'/GEN-ups-CPD/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\tt_std\n')
    for g,dg,t in sorted(gen_ups_cpd):
        o.write('%s\t%s\t%.4f\n'%(g,dg,t))

with open(out_path+'/GEN-dws-CPD/%s.tsv'%source,'w') as o:
    o.write('n1\tn2\tt_std\n')
    for g,dg,t in sorted(gen_dws_cpd):
        o.write('%s\t%s\t%.4f\n'%(g,dg,t))

sys.stderr.write('Done!\n')
