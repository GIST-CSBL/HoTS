# HoTS: Sequence based prediction of binding region and drug-target interaction.

## Introduction

Recently, many feature based drug-target interaction (DTI) prediction models are developed.
Especially, for protein feature, many models take raw amino acid sequence as the input, building end-to-end model.

This model gives some advantages for prediction, such as

  * Model catches local patterns of feature, whose information is lost in global feature. 
  * Model becomes more informative and interpretable than model using global feature

DeepConv-DTI and DeepAffinity show that deep learning model with protein sequence actually capture local residue pattern participating in interaction with ligands.
Therefore, we can hypothesize that increasing ability to capture local residue patterns will also increase performance of prediction.
Thus, how can increase ability to capture local residue patterns of drug-target prediction model?
In DeepConv-DTI, variety size of convolutional neural networks (CNN) play role of capturing local residue patterns. While the bottom layers organize queried local residue patterns to predict DTIs.
It means that training and updating weights of CNN layers to capture local residue pattern will increase performance of DTI prediction.

So, we built model on protein sequence to predict ``binding region``, which is called **Highlight on Protein Sequence (HoTS)**.
We predict ``binding regions`` of protein in the way of object detection in image processing field.

We refers ``binding region`` as consecutive amino acid residue including ``binding site`` interacting with ligand in protein-ligand complex.
By predicting ``binding region``, performance of DTI prediction increase than previous model [DeepConv-DTI](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007129).

Moreover, as pointed in [studies](https://www.researchgate.net/publication/335085389_Improved_fragment_sampling_for_ab_initio_protein_structure_prediction_using_deep_neural_networks), inter-dependency between protein moitifs must be considered for better respresentation

Our model utilized [Transformers](https://arxiv.org/abs/1706.03762) to model interdependency between sequential grids.
Moreover, we added compound token before protein grids as ``<CLS>`` token is added to predict class of sentence. Transformer also will model interaction between protein and compound.

Our model is depicted as [overview figure](Figures/Fig_1.jpg)

## Overview Figure

![OverviewFigure](Figures/Fig_1.jpg){: width="800" height="100"}

