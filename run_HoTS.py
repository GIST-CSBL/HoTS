from HoTS.model.hots import *
from HoTS.utils.build_features import *


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="""
    This Python script is used to train, validate sequence-based deep learning model for prediction of drug-target interaction (DTI) and binding region (BR)
    Deep learning model will be built by Keras with tensorflow.
    You can set almost hyper-parameters as you want, See below parameter description
    DTI, drug and protein data must be written as csv file format. And feature should be tab-delimited format for script to parse data.
    And for HoTS, Protein Sequence, binding region and SMILES are need in tsv. you can check the format in sample data. \n
    Requirement
    ============================ 
    tensorflow == 1.12.0 
    keras == 2.2.4 
    numpy 
    pandas 
    scikit-learn 
    tqdm 
    ============================\n
    contact : dlsrnsladlek@gist.ac.kr
              hjnam@gist.ac.kr\n
    """, formatter_class=argparse.RawDescriptionHelpFormatter)
    # train_params
    parser.add_argument("dti_dir", help="Training DTI information [Compound_ID, Protein_ID, Label]")
    parser.add_argument("drug_dir", help="Training drug information [Compound_ID, SMILES]")
    parser.add_argument("protein_dir", help="Training protein information [Protein_ID, Sequence]")
    parser.add_argument("hots_dir", help="Training BR information [Sequence, binding_region]")
    # validation_params
    parser.add_argument("--validation-dti-dir",  help="Test dti [Compound_ID, Protein_ID, Label]", type=str)
    parser.add_argument("--validation-drug-dir",  help="Test drug information [Compound_ID, SMILES]", type=str)
    parser.add_argument("--validation-protein-dir", help="Test Protein information [Protein_ID, Sequence]", type=str)
    parser.add_argument("--validation-hots-dir", help="Validation Binding region information [Sequence, binding_region]", type=str)
    # structure_params
    parser.add_argument("--window-sizes", '-w', help="Window sizes for model", default=[10, 15, 20, 25, 30], nargs="*", type=int)
    parser.add_argument("--n-filters", "-f", help="Number of filters for convolution layer", default=64, type=int)
    parser.add_argument("--drug-layers", '-c', help="Dense layers for drugs", default=[512, 128], nargs="*", type=int)
    parser.add_argument("--hots-dimension", '-H', help="Dimension of HoTS, D_model for transformer", default=128, type=int)
    parser.add_argument("--n-heads", help="Number of heads for multi-head attention", default=4, type=int)
    parser.add_argument("--n-transformers-hots", help="Number of transformers for BR prediction, must be less than n-transformers-dti", default=2, type=int)
    parser.add_argument("--n-transformers-dti", help="Number of transformers for DTI prediction", default=4, type=int)
    parser.add_argument("--hots-fc-layers",  help="Dense layers for concatenated layers of drug and target layer",
                        default=[256, 64], nargs="*", type=int)
    parser.add_argument("--dti-fc-layers",  help="Dense layers for concatenated layers of drug and target layer",
                        default=[256, 64], nargs="*", type=int)

    parser.add_argument("--anchors", default=[9], nargs="+", type=int, help="Basic anchors to predict BR")
    parser.add_argument("--grid-size", default=20, type=int, help="Grid size to pool protein feature from convolution results")

    # training_params
    parser.add_argument("--learning-rate", '-r', help="Learning late for training", default=1e-4, type=float)
    parser.add_argument("--n-warm-up", help="The number of warming-up epochs", default=15, type=int)
    parser.add_argument("--n-epochs", '-e', help="The number of epochs for training or validation", type=int, default=10)
    parser.add_argument("--hots-ratio", help='The number of HoTS training ratio per DTI training epoch', type=int, default=5)


    # HoTS loss params
    parser.add_argument("--retina-loss",  help="Retina loss weight", default=2., type=float)
    parser.add_argument("--confidence-loss", help="Confidence loss weight for BR prediction", default=1, type=float)
    parser.add_argument("--regression-loss", help="Regression loss weight for BR prediction", default=0.1, type=float)
    parser.add_argument("--negative-loss", help="Negative loss weight for BR prediction", default=0.1, type=float)

    # Compound Feature params
    parser.add_argument("--drug-len", "-L", help="Drug vector length", default=2048, type=int)
    parser.add_argument("--radius", "-R", help="Morgan fingerprints raidus", default=2, type=int)
    # the other hyper-parameters
    parser.add_argument("--activation", "-a", help='Activation function of model', type=str, default='gelu')
    parser.add_argument("--dropout", "-D", help="Dropout ratio", default=0.1, type=float)

    parser.add_argument("--batch-size", "-b", help="Batch size", default=16, type=int)
    parser.add_argument("--decay", "-y", help="Learning rate decay", default=1e-4, type=float)
    # output_params
    parser.add_argument("--save-model", "-m", help="Path to save model", type=str)

    args = parser.parse_args()

    prot_vec = "Sequence"
    drug_vec = "Morgan"
    protein_encoder = ProteinEncoder(prot_vec)
    compound_encoder = CompoundEncoder(drug_vec, radius=args.radius, n_bits=args.drug_len)

    hots_dimension = args.hots_dimension
    transformer_layers = [hots_dimension] * args.n_transformers_dti

    dti_model = HoTS()
    dti_model.build_model(prot_vec=prot_vec, drug_vec=drug_vec,
                          drug_layers=args.drug_layers, protein_strides=args.window_sizes, filters=args.n_filters,
                          fc_layers=args.hots_fc_layers, hots_fc_layers=args.dti_fc_layers,
                          hots_dimension=hots_dimension, protein_layers=transformer_layers,
                          n_stack_hots_prediction=args.n_transformers_hots,
                          activation='gelu', protein_encoder_config={"feature":prot_vec},
                          compound_encoder_config={"radius":args.radius, "feature":drug_vec, "n_bits":args.drug_len},
                          dropout=0.1, anchors=args.anchors, protein_grid_size=args.grid_size, hots_n_heads=args.n_heads)

    parsed_data = {}
    if args.validation_dti_dir:
        validation_dti = parse_DTI_data(args.validation_dti_dir, args.validation_drug_dir, args.validation_protein_dir,
                                        with_label=True, compound_encoder=compound_encoder, protein_encoder=protein_encoder)
        parsed_data.update({"Validation_DTI":validation_dti})
    if args.validation_hots_dir:
        validation_hots = parse_HoTS_data(args.validation_hots_dir, pdb_bound=False,
                                          compound_encoder=compound_encoder, protein_encoder=protein_encoder)
        parsed_data.update({"Validation_HoTS": validation_hots})

    parsed_data.update({
        "dti_dataset":parse_DTI_data(args.dti_dir, args.drug_dir, args.protein_dir,
                                     with_label=True, compound_encoder=compound_encoder, protein_encoder=protein_encoder)})
    training = parse_HoTS_data(args.hots_dir, pdb_bound=False,
                               compound_encoder=compound_encoder, protein_encoder=protein_encoder)
    parsed_data.update({"hots_dataset": training})
    print(parsed_data.keys())

    dti_model.training(n_epoch=args.n_epochs, batch_size=args.batch_size, hots_training_ratio=args.hots_ratio, hots_warm_up_epoch=args.n_warm_up,
                       negative_loss_weight=args.negative_loss, retina_loss_weight=args.retina_loss, conf_loss_weight=args.confidence_loss,
                       reg_loss_weight=args.regression_loss, decay=args.decay, learning_rate=args.learning_rate, **parsed_data)

    if args.save_model:
        save_dir = args.save_model
        dti_model.save_model(model_config = save_dir+".config.json",
                             hots_file    = save_dir+".HoTS.h5",
                             dti_file     = save_dir+".DTI.h5")
