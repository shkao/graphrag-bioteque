# Bioteque code

Here we provide the code used for the Bioteque resource and the pre-processings scripts used to accommodate the datasets into the knowledge graph.

**Resource**: https://bioteque.irbbarcelona.org

**Citation**: Fernandez-Torras, A., Duran-Frigola, M. Bertoni, M. Locatelli, M. & Aloy, P.
Integrating and formatting biomedical data as pre-calculated knowledge graph embeddings in the Bioteque. Nature Communications (2022). (https://doi.org/10.1038/s41467-022-33026-0)

## **How to explore an embedding space**

We provide a jupyter notebook under the path ***/demo/exploring_an_embedding_space.ipynb*** illustrating some downstream analysis one can do with an embedding space of our resource.

The notebook includes:
1. 2D (interactive) visualizations.
2. Exploration of node similarities (nearest neighbours identification).
3. Clustering the embedding space.
4. Building a predictor model to infer entity properties.

## **Programmatic access to the resource**
We provide a jupyter notebook under the path ***/demo/downloading_embeddings.ipynb*** illustrating how to programmatically download embeddings and other metadata from the resource.

The notebook includes:

1. Downloading all dataset embeddings for a given metapath or for a specific metapath-dataset of interest.
2. Access to the the list of all metapath available in our resource.
3. Access to the embedded node universe, together with some metadata (e.g. other names and synonyms) and the available metapath-dataset embeddings.

## **Pre-processing scripts**
**Important**: The datasets used in this resource have their own licenses and the use of the pre-processed data still requires compliance with the corresponding data use restriction. When using the pre-processed data of our resource, **please cite the original publication of the dataset together with our resource**.

#### **Steps**
1) Go to the folder *datasets*.  There, you will find a folder for each dataset to be processed.
2) If there is an *INSTRUCTIONS.txt* file: read it! Many datasets requires some data to be downloaded first and there you will find the instructions to do so.
3) Run the python script there: `python3 script.py`. The pre-processing dataset will be saved at *graph/raw_data/*, in the corresponding association triplet folder (depends on each dataset).
4) Once all the 'raw' versions of the datasets have been obtained, the next step is to download and prepare the different vocabulary ontologies. To do so, go to the folder *metadata/ontologies/scripts/* and follow the instructions that are detailed in the README.txt file.
5) Finally, go to the folder  *code/kgraph/* and run the script `python3 process_raw_data.py`. This script will basically process each of the raw datasets to (i) limit the nodes to the Bioteque universe, (ii) map disease terms to the reference Disease Ontology (DO) vocabulary, (iii) de/propagate the associations of those vocabularies based on the corresponding ontology, and (iv) remove redundancies in the dataset. The final processed datasets will be saved at *graph/processed/*. We used the 'propagated' version to obtain our embeddings.

#### **Disclaimers**
- We only provide the scripts to process the datasets we chose as 'reference' for our embedding resource (i.e. those available at: https://bioteque.irbbarcelona.org/sources).
- To comply with the original data licenses,  we do not provide the original data in most of the cases. Instead, we provide a script to download the data (*get_data.sh*), which is automatically called by the main python script (*script.py*). HOWEVER, some datasets requires specific licenses, accounts or are just too intricate to automate their download. As mentioned, instructions to download the data are specified in the corresponding *INSTRUCTIONS.txt* file.
- Some datasets scripts may require specific python packages for its processing. A list with all the packages used in this repository is available at *code/requirements.txt*.
- Take into account that, as many of these scripts fetch the data directly from the main repositories, changes made by the original data owners (e.g. changing the download path or the format of the files) may cause the script to fail its processing.
- Please, apart from citing our work also cite the original publication of the dataset you use.


