Data was obtained from the Genomics of Drug Sensitivity in Cancer (GDSC).

To binarize the gene expression we followed the Harmonizome pipeline (https://doi.org/10.1093/database/baw100). 
Briefly:
	--> 1. Quantile normalization
	--> 2. Scaling each Gene vector across cell lines
	--> 3. Selecting TOP 250 up- and down-regulated genes for each cell line

--> Metaedges: CLL-upr-GEN, CLL-dwr-GEN
--> URL: https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Home.html
--> Publication: https://doi.org/10.1016/j.cell.2016.06.017
