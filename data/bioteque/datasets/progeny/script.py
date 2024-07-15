#Perturbation biomarkers from pathways (Progeny, Saez-Rodriguez 2018)
import os
import sys
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps

out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
source = current_path.split('/')[-1]

#-----------Parameters-------------

#--> Manually curated mapping of pathway names to IDs to ids
pth = ['EGFR', 'Hypoxia', 'JAK.STAT', 'MAPK', 'NFkB', 'PI3K', 'TGFb', 'TNFa', 'Trail', 'VEGF', 'p53']

path_to_ids = {'EGFR':{'REACTOME':'R-HSA-177929', #Signaling by EGFR
                      'KEGG': '',
                      'WIKIPATHWAYS': 'WP1910'}, #Signaling by EGFR
               'Hypoxia':{'REACTOME':'R-HSA-2262749', # Cellular response to hypoxia
                      'KEGG':'hsa04066', #HIF-1 signaling pathway
                      'WIKIPATHWAYS':'WP2727'}, #Hypoxia response via HIF activation
               'JAK.STAT':{'REACTOME':'',
                      'KEGG':'hsa04630', #Jak-STAT signaling pathway
                      'WIKIPATHWAYS':'WP2593'}, #JAK/STAT
               'MAPK':{'REACTOME': 'R-HSA-5683057', #MAPK family signaling cascades
                      'KEGG':'hsa04010', #MAPK signaling pathway
                      'WIKIPATHWAYS':'WP422'}, #MAPK Cascade
               'NFkB':{'REACTOME':'',
                      'KEGG':'hsa04064', #NF-kappa B signaling pathway
                      'WIKIPATHWAYS':''},
               'PI3K':{'REACTOME':'R-HSA-109704', #PI3K Cascade
                      'KEGG':'hsa04151', #PI3K-Akt signaling pathway
                      'WIKIPATHWAYS':''},
               'TGFb':{'REACTOME':'R-HSA-170834', #Signaling by TGF-beta Receptor Complex
                      'KEGG': 'hsa04350', #TGF-beta signaling pathway
                      'WIKIPATHWAYS':'WP366'}, #TGF-beta signaling pathway
               'TNFa':{'REACTOME':'R-HSA-75893', #TNF signaling
                      'KEGG':'hsa04668', #TNF signaling pathway
                      'WIKIPATHWAYS':'WP231'}, #TNF signaling pathway
               'Trail':{'REACTOME':'R-HSA-75158', #TRAIL signaling
                      'KEGG':'',
                      'WIKIPATHWAYS':'WP3400'}, #TRAIL signaling
               'VEGF':{'REACTOME':'R-HSA-194138', #Signaling by VEGF
                      'KEGG':'hsa04370', #VEGF signaling pathway
                      'WIKIPATHWAYS':'WP1919'}, #Signaling by VEGF
               'p53':{'REACTOME':'',
                      'KEGG':'hsa04115', #p53 signaling pathway
                      'WIKIPATHWAYS':''}
              }

#----------------------------

#Variables
not_mapped_gene = set([])
gns =set([])
paths = set([])
it = 0
not_mapped_it = 0

#Getting mapping dict
gn2updated_gene = mps.get_gene2updatedgene()
gn2up = mps.get_gene2unip()

# Reading progeny file and writing results
o = {'PWY-upr-GEN':open(out_path+'PWY-upr-GEN/%s.tsv'%source,'w'),
     'PWY-dwr-GEN':open(out_path+'PWY-dwr-GEN/%s.tsv'%source,'w')}
for i in o:
    o[i].write('n1\tn2\tzscore\n')

with open('./progeny_matrix.csv','r') as f:
    f.readline() #header
    for l in f:
        h = l.rstrip().split('\t')
        gn = h[0]

        gns.add(gn)
        it+= len([x for x in h[1:] if x!='0'])

        try:
            gn = gn2up[gn2updated_gene[gn]]
        except KeyError:
            not_mapped_gene.add(gn)
            not_mapped_it += len([x for x in h[1:] if x!='0'])
            continue
        for ix,i in enumerate(h[1:]):
            i = float(i)
            if i != 0:
                #Iterating through all the pathway sources
                for p,pid in path_to_ids[pth[ix]].items():
                    paths.add(pid)
                    if pid != '':
                        for g in gn:
                            if i< 0:
                                o['PWY-dwr-GEN'].write('%s\t%s\t%.4f\n'%(pid,g,i))
                            else:
                                o['PWY-upr-GEN'].write('%s\t%s\t%.4f\n'%(pid,g,i))
#Clossing files
for i in o:
    o[i].close()

#Writing statistics
with open('./progeny_stats.txt','w') as o:
    o.write('* Total number of genes: %i\n'%len(gns))
    o.write('* Total number of Pathways: %i\n'%len(paths))
    o.write('* Total number of interactions: %i\n'%it)
    o.write('* Total interactions mapped: %i (%.2f%s)\n'%(it-not_mapped_it,(it-not_mapped_it)/it*100,'%'))
    o.write('* The following Genes were not mapped (%i):\n'%len(not_mapped_gene))
    o.write(';'.join(sorted(not_mapped_gene))+'\n')
sys.stderr.write('done!\n')
