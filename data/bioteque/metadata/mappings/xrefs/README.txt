In the './scripts' folder you will find the scripts used to obtain the Disease mappings to our reference vocabulariy (DO). The scripts were executed in the following order:

1) 1.get_cross_references.py
	--> Process the ontology raw files to extract the curated xrefs provided in each ontology.
	--> cross_references files will be saved at './disease'

2) 2.get_TFIDF_from_UMLS_to_xrefs_disease.py
	--> Predicts new disease xrefs by measuring the TDIDF similarity using UMLS annotations (obtained from disgenet https://www.disgenet.org/downloads). Based on the distribution of cosine distances of curated mappings, we chose a 0.5 cosine distance as a bonafide cutoff for the predictions (empirical p-value: 0.0005, check the 'Analyzing_xrefs.ipynb' notebook).

3) 3.get_doid_mappings.py
	--> Creates the file '../mappings/DIS/doid.tsv' by adding the curated xrefs + those predictions with a cosine distance < 0.5

IMPORTANT
---------
To run this scripts you first need to download the corresponding ontology files in the '../../ontologies/raw_ontologies/' folder.
To do so you have a python script ('../../ontologies/raw_ontologies/get_obos.py') that will automatically download all the files needed.
