In order to get the ontology data you need to:

1) Get your Bioontology apikey and paste it in the first line of the file './bioontology_apikey.txt'.
To get and apikey create an account here https://bioportal.bioontology.org/login, and go to 'Account Settings' via your username.

2) Run the script in this folder: 'python3 ./get_ontologies.py'

IMPORTANT
---------
The MEDDRA ontology can not be automatically downloaded as it requires a license. Therefore, 2 files will be missing:

--> ../../mappings/DIS/meddra_alt2id.tsv
--> ../raw_ontologies/meddra

This will only affect the 'offsides' dataset as it is based on MEDDRA vocabularies. Without these files, the processed dataset may not be exactly the same than the one we eventually used as we propagated the dataset using the meddra ontology before mapping to DO vocabulary. However, the mapping from MEDDRA to DO is not compromised, as we used cross references extracted from databased outside MEDDRA.

To add MEDDRA ask for a license (https://www.meddra.org/user/login), download the meddra folder and place it to the '../raw_ontologies/' path.
