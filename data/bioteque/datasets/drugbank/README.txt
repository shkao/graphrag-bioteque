Data was obtained from Drugbank.

This script generates 4 different subdatasets:

	---> 'drugbank_pd': Only keeps pharmacodynamic targets (excluding enzymes, carriers and transporters)
	- 'drugbank_pk': Only keeps pharmacokinetic targets (enzymes, carriers and transporters)
	- 'drugbank_active': Only keeps pharmacological active targets.
	- 'drugbank': Any target is kept (i.e. the union of the three above)

--> Metaedges: CPD-int-GEN
--> URL: https://go.drugbank.com/releases/latest
--> Publication: https://doi.org/10.1093/nar/gkx1037
