import numpy as np
import math
from keras.preprocessing import sequence
from multiprocessing import Pool
from keras.utils import Progbar
from model.loss import HoTSLoss
from model.Normalization import LayerNormalization, WeightNormalization, MeanOnlyBatchNormalization
from utils.metric import *
from utils.build_features import *
from utils.DataGeneratorHoTS import *
from utils.visualization import *
from keras.layers import *
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, auc, roc_curve, confusion_matrix
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2,l1, l1_l2
from keras.constraints import MinMaxNorm
from keras.preprocessing import sequence

from keras.callbacks import LearningRateScheduler, Callback

from keras.initializers import RandomNormal, Zeros, glorot_normal

class HoTS(object):

    def gelu(self, x):
        return 0.5 * x * (1 + K.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * K.pow(x, 3))))

    def PLayer(self, size, filters, activation, dropout, params_dic, norm=True, name=""):
        def f(input):
            if norm:
                model_conv = Convolution1D(filters=filters, kernel_size=size, padding='same', name=name, **params_dic)
                model_conv = WeightNormalization(model_conv, name=name+"_norm")(input)
                model_conv = MeanOnlyBatchNormalization(name=name+"_meanonly_batchnorm")(model_conv)
                #model_conv = BatchNormalization(name=name + "_batchnorm")(model_conv)
            else:
                model_conv = Convolution1D(filters=filters, kernel_size=size, padding='same', name=name, **params_dic)(input)
            model_conv = Activation(activation, name=name+"_activation")(model_conv)
            model_conv = Dropout(dropout, name=name+"_dropout")(model_conv)
            return model_conv
        return f

    def repeat_vector(self, args):
        layer_to_repeat = args[0]
        sequence_layer = args[1]
        return RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)

    def dense_norm(self, units, activation, name=None, dropout=0.0, params_dic=None, norm=True):
        def dense_func(input):
            if norm:
                densed = Dense(units, name=name, **params_dic)(input)
                densed = BatchNormalization(name=name+"_batchnorm")(densed)
                #densed = WeightNormalization(Dense(units, name=name, **params_dic), name=name+"_norm")(input)
            else:
                densed = Dense(units, name=name, **params_dic)(input)
            densed = Activation(activation, name=name+"_activation")(densed)
            densed = Dropout(dropout, name=name+"_dropout")(densed)
            return densed
        return dense_func

    def time_distribued_dense_norm(self, units, activation, name=None, dropout=0.0, params_dic=None, norm=True):
        def dense_func(input):
            if norm:
                densed = TimeDistributed(Dense(units, name=name, **params_dic), name=name+"_densed_timedistributed")(input)
                densed = TimeDistributed(LayerNormalization(name=name+"_norm"), name=name+"_timedistributed")(densed)
                #densed = TimeDistributed(WeightNormalization(Dense(units, name=name, **params_dic), name=name+"_norm"))(input)
            else:
                densed = TimeDistributed(Dense(units, name=name, **params_dic), name=name+"_densed_timedistributed")(input)
            densed = Activation(activation, name=name+"_activation")(densed)
            densed = Dropout(dropout, name=name+"_dropout")(densed)
            return densed
        return dense_func

    def attention(self, n_dims=32, dropout=0.2, name=None):
        def attn_func(query, key, value):
            dot = Dot([2,2], name=name+"_dot")([query, key])
            dot = Lambda(lambda a: a/tf.sqrt(tf.cast(K.shape(a)[-1], tf.float32)), name=name+"_scaled_dot")(dot)
            dot = Dropout(dropout)(dot)
            attn = Activation('softmax', name=name+"_softmax")(dot)
            attn = Dot([2,1], name=name+"_attended")([attn, value])
            return attn
        return attn_func

    def self_attention(self, n_dims, activation, dropout, params_dic, name="", n_heads=4):
        def self_attn_func(layer_q, layer_k, layer_v):
            #model_ph_kqvs_raw = self.dense_norm(n_dims, activation, name=name,
            #                                      dropout=dropout, params_dic=params_dic, norm=False)(protein_layer)
            n_partial_dim = int(n_dims/n_heads)
            _, _, model_partial_q_dim = K.int_shape(layer_q)
            model_partial_q_dim = int(model_partial_q_dim/n_heads)
            model_q = TimeDistributed(Dense(n_dims, activation=None, name="%s_multihead_queries"%(name), use_bias=False, ),
                                      name="%s_multihead_queries_time_distributed"%(name))(layer_q)
            model_qs = [self.crop(i*model_partial_q_dim, (i+1)*model_partial_q_dim, name="%s_query_cropped_%d"%(name, i))(model_q) for i in range(n_heads)]
            _, _, model_partial_k_dim = K.int_shape(layer_k)
            model_partial_k_dim = int(model_partial_k_dim/n_heads)
            model_k = TimeDistributed(Dense(n_dims, activation=None, name="%s_multihead_keys"%(name), use_bias=False, ),
                                      name="%s_multihead_keys_time_distributed"%(name))(layer_k)
            model_ks = [self.crop(i*model_partial_k_dim, (i+1)*model_partial_k_dim, name="%s_key_cropped_%d"%(name, i))(model_k) for i in range(n_heads)]
            _, _, model_v_dims = K.int_shape(layer_v)
            model_partial_v_dim = int(model_v_dims/n_heads)
            model_v = self.time_distribued_dense_norm(n_dims, activation, dropout=dropout, norm=True,
                                                       name="%s_multihead_values"%name, params_dic=params_dic)(layer_v)
            #model_v = Dropout(dropout)(model_v)
            #TimeDistributed(Dense(n_dims, activation=activation, name="%s_multihead_values"%(name), use_bias=True),
            #                          name="%s_multihead_values_time_distributed"%(name))(layer_v)
            #model_v = self.time_distribued_dense_norm(n_dims, activation=activation, name="%s_multihead_values"%(name), params_dic=params_dic, dropout=dropout)(layer_v)
            model_vs = [self.crop(i*model_partial_v_dim, (i+1)*model_partial_v_dim, name="%s_value_cropped_%d"%(name, i))(model_v) for i in range(n_heads)]
            model_attn = [self.attention(model_v_dims, name="%s_multihead_%d"%(name, i),
                                            dropout=dropout)(model_kqvs[0], model_kqvs[1], model_kqvs[2])
                             for i, model_kqvs in enumerate(zip(model_qs, model_ks, model_vs))]
            if len(model_attn)==1:
                model_attn = model_attn[0]
            else:
                model_attn = Concatenate()(model_attn)
            model_attn = self.time_distribued_dense_norm(n_dims, activation, dropout=dropout, norm=True,
                                                       name="%s_self_attention_dense"%name, params_dic=params_dic)(model_attn)
            model_attn = TimeDistributed(LayerNormalization(name="%s_layer_norm"),
                                         name="%s_layer_norm_values_time_distributed"%(name))(Add()([model_attn, layer_v]))
            model_attn = Dropout(dropout)(model_attn)

            return model_attn
        return self_attn_func

    def crop(self, start, end, name=""):
        def crop_func(input):
            return input[..., start:end]
        return Lambda(crop_func, name=name)

    def position_embedding(self, arg):
        input_shape = K.shape(arg)
        batch_size, seq_len, n_units = input_shape[0], input_shape[1], input_shape[2]
        #n_units = 32
        pos_input = K.tile(K.expand_dims(K.arange(0, seq_len, dtype=float), axis=0), [batch_size, 1])
        evens = K.arange(0, n_units // 2) * 2
        odds = K.arange(0, n_units // 2) * 2 + 1
        even_embd = K.sin(
            K.dot(
                K.expand_dims(pos_input, -1),
                K.expand_dims(1.0 / K.pow(
                    10000.0,
                    K.cast(evens, K.floatx()) / K.cast(n_units, K.floatx())
                ), 0)
            )
        )
        odd_embd = K.cos(
            K.dot(
                K.expand_dims(pos_input, -1),
                K.expand_dims(1.0 / K.pow(
                    10000.0, K.cast((odds - 1), K.floatx()) / K.cast(n_units, K.floatx())
                ), 0)
            )
        )
        embd = K.stack([even_embd, odd_embd], axis=-1)
        output = K.reshape(embd, [-1, seq_len, n_units])
        return output

    def modelv(self, drug_layers, protein_strides, filters, fc_layers, hots_fc_layers, dropout=0.1, n_compound_word=65,
               prot_vec="Sequence", hots_dimension=64, hots_n_heads=4,
               activation='gelu', protein_layers=None, initializer="glorot_normal", drug_len=2048, drug_vec="ECFP4",
               protein_grid_size=10, compound_grid_size=10, anchors=(10,30),  n_stack_hots_prediction=0):
        def return_tuple(value):
            if type(value) is int:
               return [value]
            else:
               return tuple(value)
        regularizer_param = 0.001
        params_dic = {"kernel_initializer": initializer,
                      # "activity_regularizer": l2(regularizer_param),
                      "kernel_regularizer": l2(regularizer_param),
        }
        if activation=='gelu':
            activation = self.gelu
        n_stack_hots_prediction = n_stack_hots_prediction
        input_d = Input(shape=(drug_len,))
        input_layer_d = input_d
        model_ds = []
        if drug_layers is not None:
            drug_layers = return_tuple(drug_layers)
            for layer_size in drug_layers:
                model_d = self.dense_norm(layer_size, activation, dropout=dropout,
                                                name="DTI_Drug_dense_%d"%layer_size,
                                                params_dic=params_dic)(input_layer_d)
                model_ds.append(model_d)
                input_layer_d = model_d

        input_p = Input(shape=(None,))
        model_p_embedding = Embedding(26, 20, name="DTI_Protein_Embedding", #embeddings_regularizer=l2(regularizer_param),
                            embeddings_initializer=initializer)(input_p)
        model_p_embedding = SpatialDropout1D(0.2)(model_p_embedding)
        embedding_inputs = model_p_embedding

        model_phs = [self.PLayer(stride_size, int(filters/2), activation, dropout, params_dic, norm=True,
                        name="HoTS_protein_conv_size_%d"%(stride_size))(embedding_inputs) for stride_size in protein_strides]
        model_ph = Concatenate()(model_phs)
        model_pts = [self.PLayer(stride_size, int(filters/2), activation, dropout, params_dic, norm=True,
                        name="DTI_protein_conv_size_%d"%(stride_size))(embedding_inputs) for stride_size in protein_strides]
        model_pt = Concatenate()(model_pts)
        model_p_interaction = Concatenate()([model_ph, model_pt])
        model_p_orig = self.time_distribued_dense_norm(hots_dimension, activation, dropout=dropout, norm=True,
                                                       name="DTI_Convolution_Dense", params_dic=params_dic)(model_p_interaction)
        model_p_orig = MaxPool1D(protein_grid_size, padding='same')(model_p_orig)
        model_p_orig = self.time_distribued_dense_norm(hots_dimension, None, dropout=dropout, norm=False,
                                        name="DTI_Protein_feature_attention_Input",params_dic={})(model_p_orig)
        #model_p_orig = MaxPool1D(protein_grid_size, padding='same')(model_p_orig)

        model_d_ref = RepeatVector(1)(model_d)
        model_d_ref = self.time_distribued_dense_norm(hots_dimension, None, dropout=dropout, params_dic={}, norm=False,
                                                        name='DTI_Drug_Representation')(model_d_ref)
        model_p_orig = Concatenate(axis=1)([model_d_ref, model_p_orig])
        model_p_pos = Lambda(self.position_embedding, output_shape=(None, hots_dimension),
              name="HoTS_Protein_Drug_Pos_embedding")(model_p_orig)
        model_p_orig = Add()([model_p_orig, model_p_pos])
        #model_p = TimeDistributed(LayerNormalization(name="DTI_Layer_norm"),
        #                                   name= "DTI_Layer_norm_time_distributed")(model_p_orig)
        model_p = model_p_orig
        model_pds = []
        model_ps = []
        #model_ps.append(model_p)
        protein_layers = return_tuple(protein_layers)
        # Attention for inter-dependency for substructure
        for z, protein_layer_size in enumerate(protein_layers):
            # Local region detection for HoTS
            if z < n_stack_hots_prediction:
                pre = "DTI_" # Ablation Study: No_Transformer_for_Binding_Region_Prediction "DTI_"
            else:
                pre = ""
            model_p_attn = self.self_attention(protein_layer_size, activation, dropout, params_dic,
                                                name=pre+'Protein_attention_%d_%d'%(protein_layer_size, z),
                                                n_heads=hots_n_heads)(model_p, model_p, model_p)
            model_p_ff = self.time_distribued_dense_norm(protein_layer_size*4, activation, name=pre+"Protein_feed_forward_%d_%d"%(protein_layer_size, z),
                                                  dropout=dropout, params_dic=params_dic, norm=True)(model_p_attn)
            #model_p_residual_input = MaxPool1D(pooling_size, padding='same')(model_p)
            model_p_ff = self.time_distribued_dense_norm(protein_layer_size, activation, dropout=dropout,
                                               name=pre+"Protein_feed_forward_%d_2"%(z),
                                             params_dic=params_dic, norm=True)(model_p_ff)
            model_p_attended = TimeDistributed(LayerNormalization(name=pre+"Protein_residual_%d"%z),
                                               name=pre+"Protein_residual_%d_time_distributed"%z)(Add()([model_p_ff, model_p_attn]))
            #model_p_attended = Add()([model_p_ff, model_p])
            model_p_attended = Dropout(dropout)(model_p_attended)
            model_p = model_p_attended
            model_pd = Lambda(lambda a: a[:, 0, :], name=pre+"DTI_representation_%d"%z)(model_p)
            model_phots = Lambda(lambda a: a[:, 1:, :], name=pre+"Protein_grid_%d"%z)(model_p)
            model_pds.append(model_pd)
            model_ps.append(model_phots)

        model_ph_residual = model_p_interaction
        model_ph_residual = MaxPool1D(protein_grid_size, padding='same', name="HoTS_Protein_residual_grid")(model_ph_residual)
        model_ph_residual = self.time_distribued_dense_norm(hots_dimension, activation, name="HoTS_Protein_resiual", dropout=dropout, params_dic=params_dic, norm=True)(model_ph_residual)

        model_ph = model_ps[n_stack_hots_prediction-1]
        model_ph = TimeDistributed(LayerNormalization(name="HoTS_Protein_last_layer_norm"),
                                   name="HoTS_Protein_last_layer_norm_time_distributed")(Add()([model_ph, model_ph_residual])) # Ablation Study: No_Transformer_for_Binding_Region_Prediction, No_Transformer : model_ph_residual
        model_hots_dense = model_ph#model_ph # model_ph_residual Ablation Study: No Transformer for BRP

        if hots_fc_layers:
            input_layer_hots = model_hots_dense
            hots_fc_layers = return_tuple(hots_fc_layers)
            for z, hots_fc_layer in enumerate(hots_fc_layers):
                model_hots_dense = self.time_distribued_dense_norm(hots_fc_layer, activation=activation,
                                                         name="HoTS_last_dense_%d_%d"%(hots_fc_layer,z), dropout=dropout,
                                                         params_dic=params_dic)(input_layer_hots)
                input_layer_hots = model_hots_dense
        model_hots = self.time_distribued_dense_norm((len(anchors))*3, name='HoTS_pooling_feature_last', dropout=0.0,
                                                     activation='sigmoid', params_dic={}, norm=False)(model_hots_dense)

        model_hots = Reshape((-1, len(anchors), 3), name='HoTS_pooling_reshape')(model_hots)

        model_p = GlobalMaxPool1D()(model_p_interaction)
        model_p = self.dense_norm(hots_dimension, activation, name="DTI_Protein_residual",
                                  dropout=dropout, params_dic=params_dic, norm=True)(model_p)

        model_d = model_ds[-1]#self.dense_norm(hots_dimension, activation, name="DTI_Drug_residual", dropout=dropout, params_dic=params_dic, norm=True)(model_ds[0])

        model_pd = self.dense_norm(hots_dimension*2, activation, name="DTI_Pharm_dense", dropout=dropout, params_dic=params_dic, norm=True)(model_pds[-1])#BatchNormalization()(model_pds[-1])
        model_t = Concatenate()([model_pd, model_p, model_d])#Concatenate()([model_pd, model_p, model_d]) # Ablation No Attention : Concatenate()([model_p, model_d]) # Ablation No Residual: Concatenate()([model_d, model_p])
        #model_t = BatchNormalization(name="DTI_Pharm_norm")(Add()([model_pd, model_t]))
        input_t = model_t
        if fc_layers is not None:
            fc_layers = return_tuple(fc_layers)
            for z, fc_layer in enumerate(fc_layers):
                model_t = self.dense_norm(fc_layer, activation, dropout=dropout,
                                                name="FC_%d"%fc_layer, params_dic=params_dic)(input_t)
                input_t = model_t
        model_t = Dense(1, name="DTI_prediction")(model_t)
        model_t = Activation('sigmoid')(model_t)
        model_hots = Model(inputs=[input_d, input_p], outputs=model_hots)
        model_f = Model(inputs=[input_d, input_p], outputs = model_t)
        return model_f, model_hots

    def __init__(self):

        self.protein_grid_size = None
        self.compound_grid_size = None
        self.anchors = None
        self.hots_dimension = None
        self.hots_n_heads = None

        self.dropout = None
        self.drug_layers = None
        self.protein_strides = None
        self.filters = None
        self.fc_layers = None
        self.hots_fc_layers = None
        self.learning_rate = None
        self.prot_vec = None
        self.drug_vec = None
        self.drug_len = None
        self.activation = None
        self.protein_layers = None
        self.protein_encoder = None
        self.compound_encoder = None
        self.reg_loss_weight = None
        self.conf_loss_weight = None
        self.negative_loss_weight = None
        self.retina_loss_weight = None
        self.decay = None
        self.model_hots = None
        self.model_t = None
        self.hots_file = None
        self.dti_file = None
        self.hots_validation_results = {}
        self.dti_validation_results = {}

    def build_model(self, dropout=0.1, drug_layers=(1024,512), protein_strides = (10,15,20,25), filters=64,
                 hots_dimension=64, n_stack_hots_prediction=0,
                 learning_rate=1e-3, decay=0.0, fc_layers=None, hots_fc_layers=None, activation="relu",
                 protein_layers=None, protein_encoder_config={}, compound_encoder_config={},
                 retina_loss_weight=2, reg_loss_weight=1., conf_loss_weight=10, negative_loss_weight=0.1,
                 protein_grid_size=10, compound_grid_size=10,  anchors=[10, 15, 20],
                 hots_n_heads=4, **kwargs):
        self.protein_grid_size = protein_grid_size
        self.compound_grid_size = compound_grid_size
        self.n_stack_hots_prediction = n_stack_hots_prediction
        self.anchors = anchors
        self.hots_dimension = hots_dimension
        self.hots_n_heads = hots_n_heads

        self.dropout = dropout
        self.drug_layers = drug_layers
        self.protein_strides = protein_strides
        self.filters = filters
        self.fc_layers = fc_layers
        self.hots_fc_layers = hots_fc_layers
        self.learning_rate = learning_rate
        self.prot_vec = protein_encoder_config["feature"]
        self.drug_vec = compound_encoder_config["feature"]
        self.drug_len = compound_encoder_config["n_bits"]
        self.activation = activation
        self.protein_layers = protein_layers
        self.protein_encoder_config = protein_encoder_config
        self.protein_encoder = ProteinEncoder(**protein_encoder_config)
        self.compound_encoder_config = compound_encoder_config
        self.compound_encoder = CompoundEncoder(**compound_encoder_config)
        self.n_compound_word = compound_encoder_config["n_compound_word"]
        self.reg_loss_weight = reg_loss_weight
        self.conf_loss_weight = conf_loss_weight
        self.negative_loss_weight = negative_loss_weight
        self.retina_loss_weight = retina_loss_weight
        self.decay = decay
        self.model_t, self.model_hots = self.modelv(self.drug_layers, self.protein_strides,
                                                    self.filters, self.fc_layers, self.hots_fc_layers, dropout=self.dropout,
                                                    prot_vec=self.prot_vec, hots_dimension=self.hots_dimension, activation=self.activation,
                                                    protein_grid_size=self.protein_grid_size, compound_grid_size=compound_grid_size,
                                                    protein_layers=self.protein_layers, drug_vec=self.drug_vec,
                                                    drug_len=self.drug_len, anchors=self.anchors, hots_n_heads=self.hots_n_heads, n_stack_hots_prediction=self.n_stack_hots_prediction)
        self.opt_hots = Adam(lr=learning_rate, decay=self.decay)
        self.opt_dti = Adam(lr=learning_rate, decay=self.decay)

        self.hots_loss = HoTSLoss(grid_size=self.protein_grid_size, anchors=self.anchors,
                                    reg_loss_weight=self.reg_loss_weight, conf_loss_weight=self.conf_loss_weight,
                                    negative_loss_weight=self.negative_loss_weight, retina_loss_weight=self.retina_loss_weight)
        self.model_t.compile(optimizer=self.opt_dti, loss='binary_crossentropy', metrics=['accuracy'])
        self.model_hots.compile(optimizer=self.opt_hots, loss=self.hots_loss.compute_hots_loss)
        print(self.__dict__)
        #K.get_session().run(tf.global_variables_initializer())

    def get_model(self):
        return self.model_hots, self.model_t

    def save_model(self, model_config=None, hots_file=None, dti_file=None):
        self.hots_file = hots_file
        self.dti_file = dti_file
        class_dict = self.__dict__
        model_t = class_dict.pop("model_t")
        model_hots = class_dict.pop("model_hots")
        if model_config is not None:
            import json
            print(class_dict)
            class_dict.pop("protein_encoder")
            class_dict.pop("compound_encoder")
            class_dict.pop("hots_loss")
            class_dict.pop('opt_dti')
            class_dict.pop('opt_hots')
            f = open(model_config, 'w')
            json.dump(class_dict, f)
            f.close()
        if hots_file is not None:
            model_hots.save(hots_file, overwrite=True)
            print("\tHoTS Model saved at %s"%hots_file)
        if dti_file is not None:
            model_t.save(dti_file, overwrite=True)
            print("\tDTI Model saved at %s"%dti_file)

    def load_model(self, model_config=None, hots_file=None, dti_file=None):
        if model_config is not None:
            import json
            f = open(model_config, encoding="UTF-8")
            class_dict = json.loads(f.read())
            f.close()
            print(class_dict)
            self.build_model(**class_dict)
            #self.summary()
            hots_file = class_dict["hots_file"]
            self.model_hots.load_weights(hots_file)
            print("\tHoTS Model loaded at %s"%hots_file)
            dti_file = class_dict["dti_file"]
            self.model_t.load_weights(dti_file)
            print("\tDTI Model loaded at %s"%dti_file)
        else:
            if hots_file is not None:
                self.model_hots.load_weights(hots_file)
                print("\tHoTS Model loaded at %s"%hots_file)
            if dti_file is not None:
                self.model_t.load_weights(dti_file)
                print("\tDTI Model loaded at %s"%dti_file)

    def summary(self, hots_plot=None, dti_plot=None):
        print("DTI summary")
        self.model_t.summary()
        #if dti_plot:
        #    from keras.utils import plot_model
        #    plot_model(self.model_t, to_file=dti_plot, show_layer_names=True)
        print("HoTS summary")
        self.model_hots.summary()
        #if hots_plot:
        #    from keras.utils import plot_model
        #    plot_model(self.model_hots, to_file=hots_plot, show_layer_names=True)

    def DTI_prediction(self, drug_feature, protein_feature, label=None, output_file=None, batch_size=32, **kwargs):
        n_steps = int(np.ceil(len(protein_feature)/batch_size))
        prediction = self.model_t.predict_generator(
                DataGeneratorDTI(drug_feature, protein_feature, protein_encoder=self.protein_encoder,
                                 compound_encoder=self.compound_encoder, batch_size=batch_size,
                                 grid_size=self.protein_grid_size, train=False), steps=n_steps)

        if output_file:
            import pandas as pd
            result_df = pd.DataFrame()
            if label:
                result_df["label"] = label
            result_df["predicted"] = prediction
            result_df.save(output_file, index=False)
        else:
            return prediction

    def DTI_validation(self, drug_feature, protein_feature, label, gamma=2, batch_size=32, threshold=None, **kwargs):
        result_dic = {}
        n_steps = int(np.ceil(len(label)/batch_size))
        prediction = self.model_t.predict_generator(
                DataGeneratorDTI(drug_feature, protein_feature, protein_encoder=self.protein_encoder,
                                 compound_encoder=self.compound_encoder, batch_size=batch_size,
                                 grid_size=self.protein_grid_size, train=False), steps=n_steps)
        '''
        for hots_layer in self.model_t.layers:
            if hots_layer.name=="DTI_representation_1":
                print(hots_layer.name)
                intermediated = Model(inputs=self.model_t.input, outputs=self.model_t.get_layer(hots_layer.name).output)
                intermediated_output = intermediated.predict([drug_feature[-3:], sequence.pad_sequences(protein_feature[-3:])])
                print(intermediated_output)
        '''
        if threshold:
            prediction_copied = prediction.copy()
            prediction_copied[prediction_copied>=threshold] = 1
            prediction_copied[prediction_copied<threshold] = 0
            tn, fp, fn, tp = confusion_matrix(label, prediction_copied).ravel()
            sen = float(tp)/(fn+tp)
            pre = float(tp)/(tp+fp)
            spe = float(tn)/(tn+fp)
            acc = float(tn+tp)/(tn+fp+fn+tp)
            f1 = (2*sen*pre)/(sen+pre)
            print("\tSen : ", sen )
            print("\tSpe : ", spe )
            print("\tPrecision : ", pre)
            print("\tAcc : ", acc )
            print("\tF1 : ", f1)
            result_dic.update({"Sen": sen, "Spe": spe, "Acc":acc, "Pre": pre, "F1": f1})
        fpr, tpr, thresholds_AUC = roc_curve(label, prediction)
        AUC = auc(fpr, tpr)
        precision, recall, thresholds = precision_recall_curve(label,prediction)
        distance = (1-fpr)**2+(1-tpr)**2
        EERs = (1-recall)/(1-precision)
        positive = sum(label)
        negative = label.shape[0]-positive
        opt_t_AUC = thresholds_AUC[np.argmin(distance)]
        opt_t_AUPR = thresholds[np.argmin(np.abs(EERs-(negative/positive)))]
        AUPR = auc(recall,precision)
        print("\tArea Under ROC Curve(AUC): %0.3f" % AUC)
        print("\tArea Under PR Curve(AUPR): %0.3f" % AUPR)
        print("\tOptimal threshold(AUC)   : %0.3f " % opt_t_AUC)
        print("\tOptimal threshold(AUPR)  : %0.3f" % opt_t_AUPR)
        result_dic.update({"AUC": AUC, "AUPR":AUPR})
        print("=================================================")
        return result_dic

    def HoTS_validation(self, drug_feature, protein_feature, index_feature, protein_names,
                        pdb_starts=None, pdb_ends=None,
                        batch_size=32, **kwargs):
        n_steps = int(np.ceil(len(protein_feature)/batch_size))
        if not pdb_starts:
            pdb_starts = [0]*len(drug_feature)
        if not pdb_ends:
            pdb_ends = [10000]*len(drug_feature)
        predicted_score, predicted_index = self.HoTS_prediction(protein_feature, drug_feature,
                                                                batch_size=batch_size)
        mean_ap = AP_calculator(index_feature, predicted_index, pdb_starts, pdb_ends,
                      min_value=10, max_value=int(self.anchors[-1]*np.e)).get_AP()
        '''
        max_len = int(np.ceil(len(protein_feature[0])/20)*20)
        protein_feature_0 = sequence.pad_sequences([protein_feature[0]], maxlen=max_len, padding='post')
        drug_feature_0 = np.stack([drug_feature[0]])
        attention_layer_0 = Model(inputs=self.model_hots.inputs, outputs=self.model_hots.get_layer(name="Protein_attention_128_0_multihead_0_softmax").output)
        attention_layer_1 = Model(inputs=self.model_hots.inputs, outputs=self.model_hots.get_layer(name="Protein_attention_128_0_multihead_1_softmax").output)
        attention_layer_2 = Model(inputs=self.model_hots.inputs, outputs=self.model_hots.get_layer(name="Protein_attention_128_0_multihead_2_softmax").output)
        attention_layer_3 = Model(inputs=self.model_hots.inputs, outputs=self.model_hots.get_layer(name="Protein_attention_128_0_multihead_3_softmax").output)
        print(attention_layer_0.predict([drug_feature_0,protein_feature_0]))
        print(attention_layer_1.predict([drug_feature_0,protein_feature_0]))
        print(attention_layer_2.predict([drug_feature_0,protein_feature_0]))
        print(attention_layer_3.predict([drug_feature_0,protein_feature_0]))
        '''
        print("\tAP : ", mean_ap)
        print("=================================================")

        return {"AP":mean_ap}


    def get_HoTS_validation(self):
        return self.hots_validation_results

    def get_DTI_validation(self):
        return self.dti_validation_results

    def HoTS_visualization(self, drug_feature, protein_feature, sequence, pdb_starts=None, pdb_ends=None, print_score=True,
                           index_feature=None, protein_names=None, line_length=100, th=0.75, batch_size=32, output_file=None, **kwargs):
        predicted_score, predicted_index = self.HoTS_prediction(protein_feature, drug_feature, th=th,
                                                                batch_size=batch_size)
        print("Prediction with %f"%th)
        if not pdb_starts:
            pdb_starts = [0]*len(drug_feature)
        if not pdb_ends:
            pdb_ends = [10000]*len(drug_feature)
        if output_file:
            output_file = open(output_file, 'w')
        if index_feature:
            for protein_name, s, i, p, pdbs, pdbe, score in zip(protein_names, sequence,
                                                                index_feature, predicted_index,
                                                                pdb_starts, pdb_ends, predicted_score):
                print(protein_name)
                p = [(pred_start, pred_end, pred_score) for pred_start, pred_end, pred_score in p
                               if ((pred_end >= pdbs) & (pred_start <= pdbe))]
                print("DTI score : ", score[0])
                print_binding(s, p, line_length, i, output_file, print_score=print_score)
        else:
            for protein_name, s, p, score in zip(protein_names, sequence, predicted_index, predicted_score):
                print(protein_name)
                print("DTI score : ", score)
                #print("HoTS Precision : ",precision, " HoTS Recall : ", recall)
                if output_file:
                    output_file.write(protein_name+'\n')
                print_binding(s, p, line_length, None, output_file, print_score=print_score)
        if output_file:
            output_file.close()

    def HoTS_train(self, protein_feature, index_feature, drug_feature, batch_size=32, **kwargs):
        train_n_steps = int((np.ceil(len(protein_feature))/batch_size))
        train_gen = DataGeneratorHoTS(protein_feature, ind_label=index_feature, ligand=drug_feature,
                              anchors=self.anchors, batch_size=batch_size,
                              train=True, shuffle=True, protein_encoder=self.protein_encoder,
                              compound_encoder=self.compound_encoder, grid_size=self.protein_grid_size)

        prog_bar = Progbar(train_n_steps)
        total_reg_loss = 0
        for i in range(train_n_steps):
            train_seq_batch, train_HoTs, hots_indice, train_ligand_batch, train_dtis, train_names = train_gen.next()
            loss = self.model_hots.train_on_batch([train_ligand_batch, train_seq_batch], train_HoTs)
            #loss_sum, reg_loss, dti_loss = loss
            reg_loss = loss
            total_reg_loss += reg_loss
            prog_bar.update(i+1, values=[("train_reg_loss", total_reg_loss/(i+1))])


    def validate_datasets(self, datasets, dataset_types=("HoTS", "DTI", "VIS"), batch_size=16):
        for dataset in datasets:
            dataset_type = dataset.split("_")[-1]
            dataset_dic = datasets[dataset]
            dataset_dic.update({"batch_size": batch_size})
            if (dataset_type=="HoTS") & (dataset_type in dataset_types):
                print("\tPrediction of " + dataset)
                self.hots_validation_results[dataset].append(self.HoTS_validation(**dataset_dic))
            elif (dataset_type=="DTI") & (dataset_type in dataset_types):
                print("\tPrediction of " + dataset)
                self.dti_validation_results[dataset].append(self.DTI_validation(**dataset_dic))
            elif (dataset_type=="VIS") & (dataset_type in dataset_types):
                self.HoTS_visualization(**dataset_dic)


    def training(self, dti_dataset, hots_dataset,n_epoch=10, batch_size=32,
                   hots_training_ratio=1, hots_warm_up_epoch=20, **kwargs):

        n_hots_trains = 0
        for dataset in kwargs:
            is_hots = dataset.split("_")[-1] == "HoTS"
            if is_hots:
                self.hots_validation_results[dataset] = []
            is_dti = dataset.split("_")[-1]=="DTI"
            if is_dti:
                self.dti_validation_results[dataset] = []

        # HoTS_data
        hots_protein_feature = hots_dataset["protein_feature"]
        index_feature =  hots_dataset["index_feature"]
        hots_drug_feature = hots_dataset["drug_feature"]

        # DTI_data
        protein_feature = dti_dataset["protein_feature"]
        drug_feature = dti_dataset["drug_feature"]
        label = dti_dataset["label"]

        # Warming-up
        for j in range(hots_warm_up_epoch):
            # Train Hots
            print("HoTS training epoch %d"%(n_hots_trains+1))
            self.HoTS_train(hots_protein_feature, index_feature, hots_drug_feature, batch_size=batch_size)
            n_hots_trains += 1
            self.validate_datasets(kwargs, dataset_types=("HoTS"), batch_size=batch_size)
        # Validate DTI
        for z in range(n_epoch):
            # Train DTI
            train_n_steps = int(np.ceil(len(label)/batch_size))

            for dti_layer in self.model_t.layers:
                if dti_layer.name.split("_")[0]=="HoTS":
                    dti_layer.trainable = False
                if dti_layer.name.split("_")[0]=="DTI":
                    dti_layer.trainable = True
                    #print("%s layer is set to non-trainable"%hots_layer.name)
            self.model_t.compile(optimizer=self.opt_dti, loss='binary_crossentropy', metrics=['accuracy'])

            history = self.model_t.fit_generator(
                    DataGeneratorDTI(drug_feature, protein_feature, train_label=label, batch_size=batch_size,
                                     protein_encoder=self.protein_encoder, compound_encoder=self.compound_encoder,
                                     grid_size=self.protein_grid_size),
                                     epochs=z+1, verbose=1, initial_epoch=z, steps_per_epoch=train_n_steps)
            # Validate DTI
            self.validate_datasets(kwargs, batch_size=batch_size)
            # Train HoTS with DTI for N times
            for dti_layer in self.model_hots.layers:
                if dti_layer.name.split("_")[0]=="HoTS":
                    dti_layer.trainable = True
                if dti_layer.name.split("_")[0]=="DTI":
                    dti_layer.trainable = False
                    #print("%s layer is set to non-trainable"%hots_layer.name)
            self.model_hots.compile(optimizer=self.opt_hots, loss=self.hots_loss.compute_hots_loss,)
            for j in range(hots_training_ratio):
                # Train Hots
                print("HoTS training epoch %d"%(n_hots_trains+1))
                self.HoTS_train(hots_protein_feature, index_feature, hots_drug_feature, batch_size=batch_size, epoch=z)
                n_hots_trains += 1
                self.validate_datasets(kwargs, batch_size=batch_size)
            for dti_layer in self.model_t.layers:
                dti_layer.trainable = True
            self.model_t.compile(optimizer=self.opt_dti, loss='binary_crossentropy', metrics=['accuracy'])
            for dti_layer in self.model_hots.layers:
                dti_layer.trainable = True
                    #print("%s layer is set to non-trainable"%hots_layer.name)
            self.model_hots.compile(optimizer=self.opt_hots, loss=self.hots_loss.compute_hots_loss)
        '''
        train_n_steps = int(np.ceil(len(label)/batch_size))
        history = self.model_t.fit_generator(
                DataGeneratorDTI(drug_feature, protein_feature, train_label=label, batch_size=batch_size,
                                 protein_encoder=self.protein_encoder, compound_encoder=self.compound_encoder,
                                 grid_size=self.protein_grid_size),
                                 epochs=z+2, verbose=1, initial_epoch=z+1, steps_per_epoch=train_n_steps)
        # Validate DTI
        self.validate_datasets(kwargs, batch_size=batch_size)
        '''

    def HoTS_prediction(self, protein_feature, drug_feature, th=0., batch_size=32, **kwargs):
        test_n_steps = int(np.ceil(len(protein_feature)/batch_size))
        test_gen = DataGeneratorHoTS(protein_feature, ind_label=None, ligand=drug_feature, name=None, anchors=self.anchors,
                                     batch_size=batch_size, train=False, shuffle=False, compound_encoder=self.compound_encoder,
                                     protein_encoder=self.protein_encoder, grid_size=self.protein_grid_size)
        predicted_dtis = []
        predicted_indice = []
        for j in range(test_n_steps):
            test_seq, test_ligand = test_gen.next()
            test_max_len = test_seq.shape[1]
            prediction_dti = self.model_t.predict_on_batch([test_ligand, test_seq])
            prediction_hots = self.model_hots.predict([test_ligand, test_seq])
            hots_pooling = HoTS_pooling(self.protein_grid_size, max_len=test_max_len, anchors=self.anchors,
                                        protein_encoder=self.protein_encoder)
            predicted_pooling_index = hots_pooling.hots_grid_to_subsequence(test_seq, prediction_hots, th=th)
            predicted_dtis.append(prediction_dti)
            predicted_indice += predicted_pooling_index
        predicted_dtis = np.concatenate(predicted_dtis)
        return predicted_dtis, predicted_indice



class HoTS_pooling(object):
    """
    Class to pool highlighted target sequence.
    Call function returns highlighted sequence on target which is used to predict class of highlighted sequence.
    """
    def __init__(self, grid_size, max_len, anchors, protein_encoder, **kwargs):
        self.grid_size = grid_size
        self.max_len = max_len
        self.anchors = anchors
        self.__protein_encoder = protein_encoder

    def round_value(self, value):
        if value < 0:
            return 0
        elif value > self.max_len:
            return int(self.max_len)
        else:
            return int(value)

    def non_maxmimal_suppression(self, hots_indice):
        hots_indice = np.array(hots_indice)
        suppressed_index_result = []
        while np.any(hots_indice):
            maximum_prediction = hots_indice[0]
            suppressed_index_result.append((int(maximum_prediction[0]), int(maximum_prediction[1]), maximum_prediction[2]))
            pop_index = []
            for i, prediction in enumerate(hots_indice):
                iou = IoU(maximum_prediction[0], maximum_prediction[1], prediction[0], prediction[1], mode='se')
                if iou >= 0.5:
                    pop_index.append(i)
            mask = np.ones_like(range(len(hots_indice)), dtype=bool)
            mask[pop_index] = False
            hots_indice = hots_indice[mask]
        return suppressed_index_result

    def hots_grid_to_subsequence(self, sequences, predicted_hots, th=0.):
        xs = predicted_hots[..., 0]
        ws = predicted_hots[..., 1]
        ys = predicted_hots[..., 2]
        index_result = []
        n_samples = xs.shape[0]
        queried_index = np.array(np.where(ys >= th)).T
        if len(queried_index)==0:
            return [[(0,0,0.00)]]*n_samples
        for sample_index, grid_index, anchor_index in queried_index:
            x = xs[sample_index, grid_index, anchor_index]*self.grid_size +\
                grid_index*self.grid_size
            w = np.exp(ws[sample_index, grid_index, anchor_index])*self.anchors[anchor_index]
            y = ys[sample_index, grid_index, anchor_index]
            hots_start = self.round_value(x - w/2.)
            hots_end = self.round_value(x + w/2.)
            #result.append(sequences[sample_index, hots_start:hots_end])
            index_result.append((sample_index, hots_start, hots_end, y))
        # Non-maximal suppression
        index_result = np.array(index_result)[np.flip(np.argsort([index[3] for index in index_result]))]

        suppressed_index_result = {i:[] for i in range(n_samples)}
        for ind in index_result:
            suppressed_index_result[int(ind[0])].append((ind[1], ind[2], ind[3]))
        suppressed_index_result = list(suppressed_index_result.values())
        pool = Pool(processes=n_samples)
        suppressed_index_result = pool.map(self.non_maxmimal_suppression, suppressed_index_result)
        pool.close()
        pool.terminate()
        pool.join()
        return suppressed_index_result

    def hots_to_subsequence(self, sequences, hots_samples):
        seq_results = []
        for hots_sample in hots_samples:
            for hots in hots_sample:
                seq_index = hots[0]
                seq_c = hots[1]
                seq_w = hots[2]
                seq_start = self.round_value(seq_c-seq_w/2.)
                seq_end = self.round_value(seq_c+seq_w/2.)
                seq = sequences[seq_index][seq_start:seq_end]
                if len(seq)>3:
                    seq_results.append(seq)
                else:
                    seq_results.append("")

        max_len = max([len(seq) for seq in seq_results])
        max_len = int(np.ceil(max_len/self.grid_size)*self.grid_size)

        seq_results = self.__protein_encoder.pad(seq_results, max_len=max_len)
        return seq_results

