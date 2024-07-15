wget -O human_string_links.txt.gz https://stringdb-static.org/download/protein.links.full.v11.0/9606.protein.links.full.v11.0.txt.gz
gunzip --force human_string_links.txt.gz

wget -O string2uniprot.tsv.gz https://string-db.org/mapping_files/uniprot/human.uniprot_2_string.2018.tsv.gz
gunzip --force string2uniprot.tsv.gz
