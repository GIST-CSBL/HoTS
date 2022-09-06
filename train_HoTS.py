
from HoTS.model.hots import *
from HoTS.utils.build_features import *
import json


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="""
    This Python script is used to train, and validate sequence-based deep learning model for prediction of drug-target interaction (DTI) and binding region (BR)
    Keras will build deep learning model with tensorflow2.
    You can set almost hyper-parameters as you want; see below parameter description.
    DTI, drug, and protein data must be written in a csv file format. And feature should be in tab-delimited format for the script to parse data.
    And for BR, Protein Sequence, binding region, and SMILES are needed in tsv. You can check the format in sample data. 

    contact : dlsrnsladlek@gist.ac.kr
              hjnam@gist.ac.kr
    
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
