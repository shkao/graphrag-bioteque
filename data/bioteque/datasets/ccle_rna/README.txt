Data was obtained from the CCLE (Depmap).

To binarize the gene expression we followed the Harmonizome pipeline (https://doi.org/10.1093/database/baw100). 
Briefly:
	--> 1. Quantile normalization
	--> 2. Scaling each Gene vector across cell lines
	--> 3. Selecting TOP 250 up- and down-regulated genes for each cell line

--> Metaedges: CLL-upr-GEN, CLL-dwr-GEN
--> URL: https://depmap.org/portal/download/all/
--> Publication: DOI: 10.1038/nature11003
