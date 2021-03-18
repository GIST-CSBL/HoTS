# HoTS: Sequence based prediction of binding regions and drug-target interactions

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

## License


HoTS follow [GPL 3.0v license](LICENSE). Therefore, HoTS is open source and free to use for everyone.

However, compounds which are found by using HoTS follows [CC-BY-NC-4.0](CC-BY-NC-SA-4.0). Thus, those compounds are freely available for academic purpose or individual research, but restricted for commecial use.

## Contact

hjnam@gist.ac.kr

dlsrnsladlek@gist.ac.kr


## Overview Figure

![OverviewFigure](Figures/Fig_1.jpg)

## Usage

```
usage: run_HoTS.py [-h] [--validation-dti-dir VALIDATION_DTI_DIR]
                   [--validation-drug-dir VALIDATION_DRUG_DIR]
                   [--validation-protein-dir VALIDATION_PROTEIN_DIR]
                   [--validation-hots-dir VALIDATION_HOTS_DIR]
                   [--window-sizes [WINDOW_SIZES [WINDOW_SIZES ...]]]
                   [--n-filters N_FILTERS]
                   [--drug-layers [DRUG_LAYERS [DRUG_LAYERS ...]]]
                   [--hots-dimension HOTS_DIMENSION] [--n-heads N_HEADS]
                   [--n-transformers-hots N_TRANSFORMERS_HOTS]
                   [--n-transformers-dti N_TRANSFORMERS_DTI]
                   [--hots-fc-layers [HOTS_FC_LAYERS [HOTS_FC_LAYERS ...]]]
                   [--dti-fc-layers [DTI_FC_LAYERS [DTI_FC_LAYERS ...]]]
                   [--anchors ANCHORS [ANCHORS ...]] [--grid-size GRID_SIZE]
                   [--learning-rate LEARNING_RATE] [--n-warm-up N_WARM_UP]
                   [--n-epochs N_EPOCHS] [--hots-ratio HOTS_RATIO]
                   [--retina-loss RETINA_LOSS]
                   [--confidence-loss CONFIDENCE_LOSS]
                   [--regression-loss REGRESSION_LOSS]
                   [--negative-loss NEGATIVE_LOSS] [--drug-len DRUG_LEN]
                   [--radius RADIUS] [--activation ACTIVATION]
                   [--dropout DROPOUT] [--batch-size BATCH_SIZE]
                   [--decay DECAY] [--save-model SAVE_MODEL]
                   dti_dir drug_dir protein_dir hots_dir

    This Python script is used to train, validate sequence-based deep learning model for prediction of drug-target interaction (DTI) and binding region (BR)
    Deep learning model will be built by Keras with tensorflow.
    You can set almost hyper-parameters as you want, See below parameter description
    DTI, drug and protein data must be written as csv file format. And feature should be tab-delimited format for script to parse data.
    And for HoTS, Protein Sequence, binding region and SMILES are need in tsv. you can check the format in sample data. 

    Requirement
    ============================ 
    tensorflow == 1.12.0 
    keras == 2.2.4 
    numpy 
    pandas 
    scikit-learn 
    tqdm 
    ============================

    contact : dlsrnsladlek@gist.ac.kr
              hjnam@gist.ac.kr

    

positional arguments:
  dti_dir               Training DTI information [Compound_ID, Protein_ID,
                        Label]
  drug_dir              Training drug information [Compound_ID, SMILES]
  protein_dir           Training protein information [Protein_ID, Sequence]
  hots_dir              Training BR information [Sequence, binding_region]

optional arguments:
  -h, --help            show this help message and exit
  --validation-dti-dir VALIDATION_DTI_DIR
                        Test dti [Compound_ID, Protein_ID, Label]
  --validation-drug-dir VALIDATION_DRUG_DIR
                        Test drug information [Compound_ID, SMILES]
  --validation-protein-dir VALIDATION_PROTEIN_DIR
                        Test Protein information [Protein_ID, Sequence]
  --validation-hots-dir VALIDATION_HOTS_DIR
                        Validation Binding region information [Sequence,
                        binding_region]
  --window-sizes [WINDOW_SIZES [WINDOW_SIZES ...]], -w [WINDOW_SIZES [WINDOW_SIZES ...]]
                        Window sizes for model
  --n-filters N_FILTERS, -f N_FILTERS
                        Number of filters for convolution layer
  --drug-layers [DRUG_LAYERS [DRUG_LAYERS ...]], -c [DRUG_LAYERS [DRUG_LAYERS ...]]
                        Dense layers for drugs
  --hots-dimension HOTS_DIMENSION, -H HOTS_DIMENSION
                        Dimension of HoTS, D_model for transformer
  --n-heads N_HEADS     Number of heads for multi-head attention
  --n-transformers-hots N_TRANSFORMERS_HOTS
                        Number of transformers for BR prediction, must be less
                        than n-transformers-dti
  --n-transformers-dti N_TRANSFORMERS_DTI
                        Number of transformers for DTI prediction
  --hots-fc-layers [HOTS_FC_LAYERS [HOTS_FC_LAYERS ...]]
                        Dense layers for concatenated layers of drug and
                        target layer
  --dti-fc-layers [DTI_FC_LAYERS [DTI_FC_LAYERS ...]]
                        Dense layers for concatenated layers of drug and
                        target layer
  --anchors ANCHORS [ANCHORS ...]
                        Basic anchors to predict BR
  --grid-size GRID_SIZE
                        Grid size to pool protein feature from convolution
                        results
  --learning-rate LEARNING_RATE, -r LEARNING_RATE
                        Learning late for training
  --n-warm-up N_WARM_UP
                        The number of warming-up epochs
  --n-epochs N_EPOCHS, -e N_EPOCHS
                        The number of epochs for training or validation
  --hots-ratio HOTS_RATIO
                        The number of HoTS training ratio per DTI training
                        epoch
  --retina-loss RETINA_LOSS
                        Retina loss weight
  --confidence-loss CONFIDENCE_LOSS
                        Confidence loss weight for BR prediction
  --regression-loss REGRESSION_LOSS
                        Regression loss weight for BR prediction
  --negative-loss NEGATIVE_LOSS
                        Negative loss weight for BR prediction
  --drug-len DRUG_LEN, -L DRUG_LEN
                        Drug vector length
  --radius RADIUS, -R RADIUS
                        Morgan fingerprints raidus
  --activation ACTIVATION, -a ACTIVATION
                        Activation function of model
  --dropout DROPOUT, -D DROPOUT
                        Dropout ratio
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        Batch size
  --decay DECAY, -y DECAY
                        Learning rate decay
  --save-model SAVE_MODEL, -m SAVE_MODEL
                        Path to save model
```


### Example command

defaults values are set as optimized parameter so you can train HoTS model with following command
 
```
python run_HoTS.py ./SampleData/DTI/Training/Training_DTI_Sample.csv \
                 ./SampleData/DTI/Training/Training_Compound_Sample.csv \ 
                 ./SampleData/DTI/Training/Training_Protein_Sample.csv \
                 ./SampleData/HoTS/Training_HoTS.tsv \ 
                 --validation-dti-dir  ./SampleData/DTI/Validation/Validation_DTI.csv \
                 --validation-drug-dir ./SampleData/DTI/Validation/Validation_Compound.csv \
                 --validation-protein-dir ./SampleData/DTI/Validation/Validation_Protein.csv \
                 --validation-hots-dir ./SampleData/HoTS/Validation_HoTS.tsv \ 
                 -m ./Saved_model
```
