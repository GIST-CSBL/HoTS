# Data Description

## Concept of binding region

In previous studies, model predicted binding sites which is residues forming bonds with ligand. 
Therefore, model predict probability to be binding site for each amino acid.
However, binding sites can vary with ligands and conformational changes induced by interaction with ligands, which make accurate and detailed prediction harder for each residue in interacting motif.
In addition, pattern recognition method like convolutional neural network (CNN)



## Binding region parsing from PockeTome

[PockeTome](https://academic.oup.com/nar/article/40/D1/D535/2903425) is a comprehensive collection of conformational ensembles of druggable sites, which can be identified experimentally from co-crystal structures in PDB.

### Concept and terminology

  * ``protein`` is and entity that is unique invariable sequence.
  * Each ``protein`` contains one or more ``domains``
  * A domain has one or more binding ``sites``, groups of amino acid residues which binds to ligands
  
Exaple of PockeTome data hierarchy :

In ABL1_HUMAN protein entity (Tyrosine-protein kinase ABL1)

  * SH2 domain
  * SH3 domain
  * Protein kinase domain
    - ATP sites
    - Myristoryl sites
  
### Pocketome entry organization

PockeTome use ``siteFinder`` algorithm that automatically collects, clusters, analyzes and validates the binding pocket structures based on consistency of their composition and spatial configuration between multiple members.

During process, highly homologous protein (94% sequence identity) can be merged into single entry.

Pocketome entry satisfies 3 criteria

  * Its protein should be reviewed in UniProt
  * It has co-crystallized complex with at lease one drug-like ligand
  * it should be presented in at least two PDB entries 

## Binding region parsing from scPDB

This tsv file is binding region result parsed from [scPDB](http://bioinfo-pharma.u-strasbg.fr/scPDB/)

Each scPDB entry contains 3 files

  * ``protein.mol2``: coordination of protein in 3D-complex
  * ``ligand.mol2``: coordination of ligand in 3D-complex
  * ``site.mol2``: coordination of binding site in 3D-complex structure
  
We parsed ``mol2`` file with ``biopandas`` module, providing dataframe of each structure.

## Statistics

As a result, we constructed three dataset (training, validation, independent test) by parsing binding regions from binding sites. 
The number of total binding regions decreased respect to the number of binding sites for all datasets by merging binding sites, where as the total length of binding regions are approximately doubled respect to the number of binding sites.

|                                    |   Training dataset<br>(Pocketome human)   |  Validation dataset<br>(Pocketome mouse)  | Test dataset<br>(scPDB with distinct proteins) |
|------------------------------------|:-----------------------------------------:|:-----------------------------------------:|:----------------------------------------------:|
|    # of entries                    |                   11922                   |                    876                    |                           1832                 |
|    # of proteins                   |                    1003                   |                    103                    |                           1194                 |
|    # of binding sites              |                   379780                  |                   24708                   |                           52029                |
|    Average % of binding sites      |                   7.04%                   |                   8.71%                   |                           8.62%                |
|    # of binding regions            |                   81356                   |                   6382                    |                          12141                 |
|    Total length of binding regions |                  677679                   |                  46193                    |                           84632                |
|    Average % of binding regions    |                   14.09%                  |                   17.17%                  |                           15.56%               |