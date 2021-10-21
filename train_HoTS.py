
from HoTS.model.hots import *
from HoTS.utils.build_features import *
import json


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="""
    This Python script is used to train, validate sequence-based deep learning model for prediction of drug-target interaction (DTI) and binding region (BR)
    Deep learning model will be built by Keras with tensorflow.
    You can set almost hyper-parameters as you want, See below parameter description
    DTI, drug and protein data must be written as csv file format. And feature should be tab-delimited format for script to parse data.
    And for HoTS, Protein Sequence, binding region and SMILES are need in tsv. you can check the format in sample data. \n
    Requirements
    ============================================================================================
    tensorflow == 1.12.0 
    keras == 2.2.4 
    numpy 
    pandas 
    scikit-learn 
    tqdm 
    rdkit
    ============================================================================================\n
    contact : dlsrnsladlek@gist.ac.kr
              hjnam@gist.ac.kr\n

    Input configuration

    ============================================================================================

    ## Input file paramters

    "dti_dir"\t\t\t: Training DTI file path
    "drug_dir"\t\t\t: Training Compound file path
    "protein_dir"\t\t: Training Protein file path
    "hots_dir"\t\t\t: Training BR file path
    "validation_dti_dir"\t: Validation file path
    "validation_drug_dir"\t: Validaton file path
    "validation_protein_dir"\t: Validation file path
    "validation_hots_dir"\t: Validation BR file path

    ## Compound feature paramters
    
    "drug_len"\t\t: the number of bits for Morgan fingerprint
    "radius"\t\t\t" the size of radius for Morgan fingerprint

    ## Model shape parameters

    "window_sizes"\t\t: Protein convolution window sizes (should be list of integers)
    "n_filters"\t\t\t: Convolution filter size
    "drug_layers"\t\t: Dense layers on compound fingerprint (should be list of integers)
    "hots_dimension"\t\t: Size of dimension for Transformer
    "n_heads"\t\t\t: the number of heads in Transformer
    "n_transformers_hots"\t: the number of Transformer blocks for BR prediction
    "n_transformers_dti"\t: the number of Transformer blocks for DTI prediction
    "hots_fc_layers"\t\t: Dense layers for BR prediction (should be list of integers)
    "dti_fc_layers"\t\t: Dense layers for DTI prediction (should be list of integers)
    "anchors"\t\t\t: Predifined widths (anchor without coord offset, should be list of integers)
    "grid_size"\t\t\t: Protein grid size

    ## Training parameters

    "learning_rate"\t\t: Learning rate
    "n_pretrain"\t\t: the number of BR pre-training epochs
    "n_epochs"\t\t: the number of DTI training epochs
    "hots_ratio"\t\t: the number of BR training epochs per one DTI training
    "activation"\t\t: activation function of model
    "dropout"\t\t\t: Dropout rate
    "batch_size"\t\t: Training mini-batch size
    "decay"\t\t\t: Learning rate decay

    ## Loss parameters
    "retina_loss"\t\t: Retina loss weight
    "confidence_loss"\t\t: Confidence loss weight for BR prediction
    "regression_loss"\t\t: Regression loss weight for BR prediction
    "negative_loss"\t\t: Negative loss eight for BR prediction
    

    ## Output paramters
    "output"\t\t\t: Output file path, this script will result in 
\t\t\t\t\t{output}.config.json\t: Model hyperparameter file
\t\t\t\t\t{output}.HoTS.h5\t: BR prediction model weight file
\t\t\t\t\t{output}.DTI.h5\t: DTI prediction model weight file
    
    """, formatter_class=argparse.RawDescriptionHelpFormatter)
    # train_params
    parser.add_argument("input_config", help="input configuration json file")
    args = parser.parse_args()

    with open(args.input_config) as input_config:
         args = json.load(input_config)#vars(parser.parse_args())

    prot_vec = "Sequence"
    drug_vec = "Morgan"
    protein_encoder = ProteinEncoder(prot_vec)
    compound_encoder = CompoundEncoder(drug_vec, radius=args["radius"], n_bits=args["drug_len"])

    hots_dimension = args["hots_dimension"]
    transformer_layers = [hots_dimension] * args["n_transformers_dti"]

    dti_model = HoTS()
    dti_model.build_model(prot_vec=prot_vec, drug_vec=drug_vec,
                          drug_layers=args["drug_layers"], protein_strides=args["window_sizes"], filters=args["n_filters"],
                          fc_layers=args["hots_fc_layers"], hots_fc_layers=args["dti_fc_layers"],
                          hots_dimension=hots_dimension, protein_layers=transformer_layers,
                          n_stack_hots_prediction=args["n_transformers_hots"],
                          activation='gelu', protein_encoder_config={"feature":prot_vec},
                          compound_encoder_config={"radius":args["radius"], "feature":drug_vec, "n_bits":args["drug_len"]},
                          dropout=0.1, anchors=args["anchors"], protein_grid_size=args["grid_size"], hots_n_heads=args["n_heads"])

    parsed_data = {}
    if "validation_dti_dir" in args.keys():
        validation_dti = parse_DTI_data(args["validation_dti_dir"], args["validation_drug_dir"], args["validation_protein_dir"],
                                        with_label=True, compound_encoder=compound_encoder, protein_encoder=protein_encoder)
        parsed_data.update({"Validation_DTI":validation_dti})
    if "validation_hots_dir" in args.keys():
        validation_hots = parse_HoTS_data(args["validation_hots_dir"], pdb_bound=False,
                                          compound_encoder=compound_encoder, protein_encoder=protein_encoder)
        parsed_data.update({"Validation_HoTS": validation_hots})

    parsed_data.update({
        "dti_dataset":parse_DTI_data(args["dti_dir"], args["drug_dir"], args["protein_dir"],
                                     with_label=True, compound_encoder=compound_encoder, protein_encoder=protein_encoder)})
    training = parse_HoTS_data(args["hots_dir"], pdb_bound=False,
                               compound_encoder=compound_encoder, protein_encoder=protein_encoder)
    parsed_data.update({"hots_dataset": training})

    dti_model.training(n_epoch=args["n_epochs"], batch_size=args["batch_size"], hots_training_ratio=args["hots_ratio"], hots_warm_up_epoch=args["n_pretrain"],
                       negative_loss_weight=args["negative_loss"], retina_loss_weight=args["retina_loss"], conf_loss_weight=args["confidence_loss"],
                       reg_loss_weight=args["regression_loss"], decay=args["decay"], learning_rate=args["learning_rate"], **parsed_data)

    if "output" in args.keys():
        save_dir = args["output"]
        dti_model.save_model(model_config = save_dir+".config.json",
                             hots_file    = save_dir+".HoTS.h5",
                             dti_file     = save_dir+".DTI.h5")
