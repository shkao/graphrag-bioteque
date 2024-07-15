
Data was obtained from the publication: "Predicting Drug Response and Synergy Using a Deep Learning Model of Human Cancer Cells "

DrugCell AUC data ('drugcell_auc.tsv') was obtained by combining the files 'drugcell_train.txt', 'drugcell_test.txt' and 'drugcell_val.txt' from: https://github.com/idekerlab/DrugCell/tree/public/data
Drug and cell ids were mapped to Drugcell ids using the mapping files 'cell2ind.txt' and 'drug2ind.txt' available in the same github.

To binarize the drug sensitivty we implemented the waterfall method (as first described elsewhere DOI: 10.1038/nature11003).
We required at least an AUC <0.9, and minimum of 1% and maximum of 20% sensitive cell lines for each drug.


--> Metaedges: CLL-sns-CPD
--> URL: https://doi.org/10.1016/j.ccell.2020.09.014
--> Publication: https://doi.org/10.1016/j.ccell.2020.09.014

