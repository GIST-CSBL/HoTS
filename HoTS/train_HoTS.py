import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from model.HoTS import *
from utils.build_features import *
import math
import gc
import keras.backend as K

# Set GPU occupy
'''
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
'''

prot_vec = "Sequence"
pssm_json = "/home/dlsrnsi/DTI/HoTS/Data/PSSM.json"
drug_vec = "Morgan"
drug_len = 2048
radius = 2
protein_encoder = ProteinEncoder(prot_vec)
compound_encoder = CompoundEncoder(drug_vec, radius=radius, n_bits=drug_len)
th = 0.5

parsed_data = {}
training = parse_HoTS_data("/home/dlsrnsi/DTI/HoTS/Data/DataSet/training_HoTS.tsv", pdb_bound=True,
                                     compound_encoder=compound_encoder, protein_encoder=protein_encoder)
parsed_data.update({"hots_dataset":training})

validation_data_homologous = parse_HoTS_data("/home/dlsrnsi/DTI/HoTS/Data/DataSet/validation_HoTS.tsv",
                               pdb_bound=True, compound_encoder=compound_encoder, protein_encoder=protein_encoder)
validation_data_homologous.update({"report_micro":False})

validation_data_matador = parse_DTI_data("/DAS_Storage1/Drug_AI_project/validation_dataset/validation_dti.csv",
           "/DAS_Storage1/Drug_AI_project/validation_dataset/validation_compound.csv",
           "/DAS_Storage1/Drug_AI_project/validation_dataset/validation_protein.csv",
           with_label=True, prot_len=2500, prot_vec=prot_vec, drug_vec=drug_vec, drug_len=drug_len,
                                         compound_encoder=compound_encoder, protein_encoder=protein_encoder)

test_dic = {
            "MATADOR_DTI": validation_data_matador,
            "validation_HoTS":validation_data_homologous
            }
parsed_data.update(test_dic)

for number in range(10):
    parsed_data.update({
        "dti_dataset":parse_DTI_data("/DAS_Storage1/Drug_AI_project/DTI_dataset/Training_Dataset_vertebrate/%d.csv"%number,
       "/DAS_Storage1/Drug_AI_project/DTI_dataset/DTI/Compounds.csv",
       "/DAS_Storage1/Drug_AI_project/DTI_dataset/DTI/Proteins.csv",
       with_label=True, prot_len=2500, prot_vec=prot_vec, drug_vec=drug_vec, drug_len=drug_len,
                                     compound_encoder=compound_encoder, protein_encoder=protein_encoder)})
    dti_model = HoTS()

    dti_model.build_model(prot_vec=prot_vec, drug_vec=drug_vec,
                          drug_layers=[512, 128], protein_strides=[10, 15, 20, 25, 30], filters=128,
                          fc_layers=[256, 64], hots_fc_layers=[256, 64],
                          hots_dimension=128, protein_layers=[128, 128, 128, 128], n_stack_hots_prediction=2,
                          decay=0.0001, learning_rate=0.0001, activation='gelu', protein_encoder_config={"feature":prot_vec},
                          compound_encoder_config={"radius":radius, "feature":drug_vec, "n_bits":drug_len, "n_compound_word":65},
                          dropout=0.05, negative_loss_weight=0.1, retina_loss_weight=2,
                      conf_loss_weight=1, reg_loss_weight=0.1,
                      anchors=[10], protein_grid_size=20, hots_n_heads=4)
    #dti_model.summary()
    dti_model.HoTS_validation(**validation_data_homologous) # Baseline for BR prediction
    dti_model.training(n_epoch=10, batch_size=32, hots_training_ratio=0, hots_warm_up_epoch=0, **parsed_data)

    dti_model.save_model(model_config = "/home/dlsrnsi/DTI/HoTS/Model_new/%d_HoTS_config_9_No_binding_region_training.json"%number,
                         hots_file    = "/home/dlsrnsi/DTI/HoTS/Model_new/%d_HoTS_9_No_binding_region_training.h5"%number,
                         dti_file     = "/home/dlsrnsi/DTI/HoTS/Model_new/%d_DTI_9_No_binding_region_training.h5"%number)
    del dti_model
    gc.collect()
    K.clear_session()
