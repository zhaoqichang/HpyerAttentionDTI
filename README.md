# HpyerAttentionDTI
HyperAttentionDTI: improving drug–protein interaction prediction by sequence-based deep learning with attention mechanism
This repository contains the source code and the data.

## HpyerAttentionDTI

<div align="center">
<p><img src="model.jpg" width="600" /></p>
</div>

## Setup and dependencies 

Dependencies:
- python 3.6
- pytorch >=1.2
- numpy
- sklearn
- tqdm
- tensorboardX
- prefetch_generator

## Resources:
+ README.md: this file.
+ data: The datasets used in paper.
	+ DrugBank.txt:  
	+ KIBA.txt: 
	+ Davis.txt
	In the directory of data, we now have the original data "DrugBank/KIBA/Davis.txt" as follows:

	```
	Drug_ID Protein_ID Drug_SMILES Amino_acid_sequence interaction
	DB00303 P45059 [H][C@]12[C@@H]... MVKFNSSRKSGKSKKTIRKLT... 1
	DB00114 P19113 CC1=NC=C(COP(O)... MMEPEEYRERGREMVDYICQY... 1
	DB00117 P19113 N[C@@H](CC1=CNC... MMEPEEYRERGREMVDYICQY... 1
	...
	...
	...
	DB00441 P48050 NC1=NC(=O)N(C=C... MHGHSRNGQAHVPRRKRRNRF... 0
	DB08532 O00341 FC1=CC=CC=C1C1=... MVPHAILARGRDVCRRNGLLI... 0

	```
+ dataset.py: data process.
+ HpyerAttentionDTI_main.py: train and test the model.
+ hyperparameter.py: set the hyperparameter of HpyerAttentionDTI
+ model.py: HpyerAttentionDTA model architecture
+ pytorchtools: early stopping



# Run:

python HpyerAttentionDTI_main.py
