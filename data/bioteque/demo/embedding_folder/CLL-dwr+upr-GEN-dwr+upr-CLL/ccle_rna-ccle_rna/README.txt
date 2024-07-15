>metapath	CLL-dwr+upr-GEN-dwr+upr-CLL
>metapath_desc	"CELL-down/up_reg.-GENE-down/up_reg.-CELL"
>datasets
	1.CLL-dwr+upr-GEN --> ccle_rna
	2.GEN-dwr+upr-CLL --> ccle_rna

>#CLL	1198

>network.h5
    'edges'--> [CLL_ids, CLL_ids] matrix where 'ids' are the node position in 'CLL_ids.txt' and 'CLL_ids.txt'.
    'weights' --> DWPC weight of the edge.

>analytical_card.png
"Analytical card describing the embedding space"

    Legend
    · · · · · · · · · ·
    ·           · · B ·
    ·           · · · ·
    ·     A     · · C ·
    ·           · · · ·
    ·           · · D ·
    · · · · · · · · · ·
    · · · · · · ·
    ·     E     ·
    · · · · · · ·

    A) 2D projection of the embedding space. Each dot is a node (colored by entity).
    B) Preservation of the original metapath network by the embedding space.
    C) Cosine distance distribution between the node embeddings.
    D) Euclidean distance distribution between the node embeddings.
    E) Top biological associations recapitulated by the embedding space.
    * They grey color is used when a measure involves both the source and target entities.

