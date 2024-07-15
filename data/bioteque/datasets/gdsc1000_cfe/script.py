import sys
import os
import subprocess
import numpy as np
import pandas as pd
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#--Downloading the data
subprocess.Popen('./get_data.sh', shell = True).wait()

#Mapings
df = pd.read_excel('./Cell_Lines_Details.xlsx')
cl2n = dict(zip([str(int(x)) for x in df['COSMIC identifier'] if not pd.isnull(x)],list(df['Sample Name'].astype(str))))
cl2id = dict(zip(list(cl2n.values()),mps.cl2ID(list(cl2n.values()))))
g2up = mps.get_gene2unip()

p = './CellLines_Mo_BEMs/'
m_mut, m_cnd, m_cnu, m_mth = set([]), set([]), set([]), set([])
for file in os.listdir(p):
    with open(p+file,'r') as f:
        cls = np.asarray(f.readline().rstrip('\n').split('\t')[4:])
        for l in f:
            h = np.asarray(l.rstrip().split('\t'))
            if h[0].endswith('_mut'):
                g = h[0][:-4]
                if g not in g2up: continue
                gs = g2up[g]
                cl = cls[np.where(h[4:]=='1')[0]]

                for c in cl:
                    if c not in cl2n: continue
                    c = cl2id[cl2n[c]]
                    if c == None: continue
                    for g in gs:
                        m_mut.add((c,g))

            elif (h[0].startswith('loss') or h[0].startswith('gain')) and h[0].rstrip().endswith(')'):

                GS = h[0].split('(')[-1].split(')')[0].split(',')
                for G in GS:
                    if G not in g2up: continue
                    gs = g2up[G]
                    cl = cls[np.where(h[4:]=='1')[0]]

                    for c in cl:
                        if c not in cl2n: continue
                        c = cl2id[cl2n[c]]
                        if c == None: continue
                        for g in gs:
                            if h[0].startswith('loss'):
                                m_cnd.add((c,g))
                            elif h[0].startswith('gain'):
                                m_cnu.add((c,g))
            
            elif h[0].endswith('_HypMET'):
                g = h[0].split('(')[-1].split(')')[0]
                if g not in g2up: continue
                gs = g2up[g]
                cl = cls[np.where(h[4:]=='1')[0]]

                for c in cl:
                    if c not in cl2n: continue
                    c = cl2id[cl2n[c]]
                    if c == None: continue
                    for g in gs:
                        m_mth.add((c,g))

m_mut = np.asarray(sorted(m_mut,key=lambda x: x[0]+x[1]))
m_cnd = np.asarray(sorted(m_cnd,key=lambda x: x[0]+x[1]))
m_cnu = np.asarray(sorted(m_cnu,key=lambda x: x[0]+x[1]))
m_mth = np.asarray(sorted(m_mth,key=lambda x: x[0]+x[1]))

#Mutation
with open(out_path+'/CLL-mut-GEN/%s.tsv'%(source),'w') as o:
    o.write('n1\tn2\n')
    for r in m_mut:
        o.write('%s\t%s\n'%(r[0],r[1]))
#CND
with open(out_path+'/CLL-cnd-GEN/%s.tsv'%(source),'w') as o:
    o.write('n1\tn2\n')
    for r in m_cnd:
        o.write('%s\t%s\n'%(r[0],r[1]))

#CNU
with open(out_path+'/CLL-cnu-GEN/%s.tsv'%(source),'w') as o:
    o.write('n1\tn2\n')
    for r in m_cnu:
        o.write('%s\t%s\n'%(r[0],r[1]))
        
#Meth
with open(out_path+'/CLL-mth-GEN/%s.tsv'%(source),'w') as o:
    o.write('n1\tn2\n')
    for r in m_mth:
        o.write('%s\t%s\n'%(r[0],r[1]))

sys.stderr.write('Done!\n')

