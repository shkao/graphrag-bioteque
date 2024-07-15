import sys
import os
import numpy as np
import copy
from tqdm import tqdm_notebook as tqdm
sys.path.insert(0, '../../code/kgraph/utils/')
import mappers as mps
import harm_utils as utls
out_path = '../../graph/raw/'
current_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])

source = current_path.split('/')[-1]
def generate_graph_csv(db,dt,metaedge,attribute_index=3,attribute=None,map_attribute=False):
    """
    Downloads the data and writes the csv for the neo4the_atrj graph.

    db -- database id according to Harmonizome
    dt -- data type id according to Harmoniuzome
    metaedge -- str/list with the desired metaedges from this data. If the metaedge is a list, the output write should be specified manually.
    attribute_index -- Where the attribute is in the file. If a list of index is given, it will concatenate them using '|'
    attribute -- Attribute to be mapped.
    """

    #Variables
    w2mp = {1:'CLL-cnu-GEN',-1:'CLL-cnd-GEN'} #Added manually

    # 1) Download de network
    sys.stderr.write('Downloading the dataset...')
    utls.download_datasets([(current_path,db)],[dt],decompress=True)
    sys.stderr.write('done!\n')

    # 2) Reading file and build matrix
    sys.stderr.write('Reading the dataset...')
    try:
        mat = utls.file2matrix(current_path+'/%s'%dt[:-3],attribute_index=attribute_index)
    except FileNotFoundError:
        sys.exit('\nUps... It seems that the input file is not downloaded. Remember to uncomment the line 33...')
    sys.stderr.write('done!\n')

    #Keeping information for measuring some stats...
    genes = set(mat[:,0])
    if attribute == 'Gene':
        genes = list(genes | set(mat[:,1]))
    else:
        genes = list(genes)
    atr = copy.copy(mat[:,1])
    it_count = len(mat)
    mapp_it_count = 0 # I will count each interaction that is written in the end
    atr_unmapped = set([])
    gns_unmmaped = set([])

    # 3) Mapping

    #3.1 Getting geneID2uniprot dictionary. I will map the genes in situ, when writing the file (later)
    sys.stderr.write('Mapping proteins...')
    gn2up = mps.get_geneID2unip()

    #3.2 Mapping the attribute
    if map_attribute and attribute != 'Gene':
        mat[:,1] = mps.mapping(mat[:,1],attribute)
        atr_unmapped = set(atr[np.where(mat[:,1] == None)[0]]) # stats...

    #4) Writting files
    sys.stderr.write('Writing files...')
    if type(metaedge) == list:
        o = {mp:open(out_path+mp+'/%s.tsv'%source,'w') for mp in metaedge}
    else:
        o = {metaedge:open(out_path+metaedge+'/%s.tsv'%source,'w')}

    #4.1 Writing header
    for i in o:
        o[i].write('n1\tn2\n')

    c = -1
    for r in mat:
        c+=1
        g,atr_value,w = str(r[0]),r[1],r[2] #gene, atr, weight (1/-1)
        
        if g not in gn2up:
            gns_unmmaped.add(g) # stats...
            continue
        #Ignoring unmapped atr
        if atr_value is None:
            continue

        #Mapping gene in situ
        ups = gn2up[g]
        if len(ups) > 0:
            mapp_it_count+=1 # stats...
            for up in ups:
                o[w2mp[w]].write('%s\t%s\n'%(atr_value,up))

        else:
            gns_unmmaped.add(g) # stats...
            continue

    #Writing stats
    with open('./stats.txt','w') as output_stats:
        output_stats.write('* Total number of Genes: %i\n'%len(set(genes)))
        output_stats.write('* Total number of %ss: %i\n'%(attribute,len(set(atr))))
        output_stats.write('* Total number of interactions: %i\n'%it_count)
        output_stats.write('* Total interactions mapped: %i (%.2f%s)\n'%(mapp_it_count,mapp_it_count/it_count*100,'%'))
        output_stats.write('* The following %ss were not mapped (%i):\n'%(attribute,len(atr_unmapped)))
        output_stats.write(';'.join(sorted(set(atr_unmapped)))+'\n')
        output_stats.write('* The following Genes were not mapped (%i):\n'%(len(gns_unmmaped)))
        output_stats.write(';'.join(sorted(set(gns_unmmaped)))+'\n')

    sys.stderr.write('done!\n')

#-------------------------------------------------------------------------------------------
#RUN
metaedge = ['CLL-cnu-GEN','CLL-cnd-GEN']
db = 'cclecnv'
dt = 'gene_attribute_edges.txt.gz'
idx = [3]
attribute = 'Cell'
map_attribute = True
generate_graph_csv(db,dt,metaedge,attribute_index=idx,attribute=attribute,map_attribute = map_attribute)
