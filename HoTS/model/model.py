import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow_addons.layers import WeightNormalization

import math


class HoTSModel(object):

    def __init__(self, drug_layers, protein_strides, filters, fc_layers, hots_fc_layers, dropout=0.1,
                 hots_dimension=64, hots_n_heads=4, activation='gelu', protein_layers=None,
                 initializer="glorot_normal", drug_len=2048,
                 protein_grid_size=10, anchors=(10,30),  n_stack_hots_prediction=0, **kwargs):
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

        # Drug input embedding
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

        # Protein input embedding
        input_p = Input(shape=(None,))
        model_p_embedding = Embedding(26, 20, name="DTI_Protein_Embedding", #embeddings_regularizer=l2(regularizer_param),
                            embeddings_initializer=initializer)(input_p)
        model_p_embedding = SpatialDropout1D(0.2)(model_p_embedding)
        embedding_inputs = model_p_embedding

        # Conv layers for protein grid encoding
        model_phs = [self.PLayer(stride_size, int(filters/2), activation, dropout, params_dic, norm=True,
                        name="HoTS_protein_conv_size_%d"%(stride_size))(embedding_inputs) for stride_size in protein_strides]
        model_ph = Concatenate()(model_phs)
        #model_ph_gate = Activation('sigmoid')(model_ph)
        model_pts = [self.PLayer(stride_size, int(filters/2), activation, dropout, params_dic, norm=True,
                        name="DTI_protein_conv_size_%d"%(stride_size))(embedding_inputs) for stride_size in protein_strides]
        model_pt = Concatenate()(model_pts)
        #model_pt_gate = Activation("sigmoid")(model_pt)
        model_p_interaction = Concatenate()([model_ph, model_pt])
        #model_p_interaction = Multiply()([model_ph, model_pt_gate])
        model_p_orig = model_p_interaction

        # Convert to transformer input
        model_p_orig = self.time_distributed_dense_norm(hots_dimension, activation, dropout=dropout, norm=True,
                                                       name="DTI_Convolution_Dense", params_dic=params_dic)(model_p_orig)
        model_p_orig = MaxPool1D(protein_grid_size, padding='same')(model_p_orig)

        model_p_orig = self.time_distributed_dense_norm(hots_dimension, None, dropout=0.0, norm=True, params_dic=params_dic,
                                                        use_bias=False, name="DTI_Protein_feature_attention_Input")(model_p_orig)
        model_d_ref = RepeatVector(1)(model_ds[-1])
        model_d_ref = self.time_distributed_dense_norm(hots_dimension, None, dropout=0.0, norm=True, params_dic=params_dic,
                                                       use_bias=False, name='DTI_Drug_Representation')(model_d_ref)
        model_p_orig = Concatenate(axis=1)([model_d_ref, model_p_orig])
        '''
        model_d_ref = RepeatVector(1)(model_ds[-1])
        model_p_orig = Concatenate(axis=1)([model_d_ref, model_p_orig])
        model_p_orig = self.time_distributed_dense_norm(hots_dimension, None, dropout=0.0, norm=False, params_dic=params_dic,
                                                       name='DTI_Pharm_Representation')(model_p_orig)
        '''
        # Positional embedding
        model_p_pos = Lambda(self.position_embedding, output_shape=(None, hots_dimension),
              name="HoTS_Protein_Drug_Pos_embedding")(model_p_orig)
        model_p_orig = Add()([model_p_orig, model_p_pos])
        #model_p_orig = Concatenate(axis=1)([model_d_ref, model_p_orig])
        #model_p_orig = Dropout(dropout)(model_p_orig)

        # Mask preparing
        input_mask = Input(shape=(None,))
        attention_mask = Lambda(self.repeat_vector, output_shape=(None, None,))([input_mask, input_mask])

        # Transformers
        model_p = model_p_orig
        model_pds = []
        model_ps = []
        protein_layers = return_tuple(protein_layers)

        for z, protein_layer_size in enumerate(protein_layers):
            pre = "DTI_"

            # Self-attention
            model_p_input = LayerNormalization(name=pre + "attention_input_%d" % z)(model_p)
            model_p_attn = self.self_attention(protein_layer_size, activation, dropout, params_dic,
                                                name=pre+'Protein_attention_%d_%d'%(protein_layer_size, z),
                                                n_heads=hots_n_heads)(model_p_input, model_p_input, model_p_input, attention_mask)
            model_p_attn = Add(name=pre + "attention_added_%d" % z)([model_p, model_p_attn])

            # Position-wise Feed-forward
            model_p_ff = self.time_distributed_dense_norm(protein_layer_size*4, activation,
                                                         name=pre+"Protein_feed_forward_%d_%d"%(protein_layer_size, z),
                                                  dropout=dropout, params_dic=params_dic, norm=True)(model_p_attn)
            model_p_ff = self.time_distributed_dense_norm(protein_layer_size, None, dropout=dropout, norm=False,
                                               name=pre+"Protein_feed_forward_%d_2"%(z),
                                             params_dic=params_dic)(model_p_ff)
            model_p_attended = Add(name=pre + "feed_forward_added_%d" % z)([model_p_ff, model_p_attn])
            model_p_attended = Dropout(dropout)(model_p_attended)
            model_p = model_p_attended
            # Compound token
            model_pd = Lambda(lambda a: a[:, 0, :], name=pre+"DTI_representation_%d"%z)(model_p)
            model_pds.append(model_pd)
            # Protein encoding
            model_phots = Lambda(lambda a: a[:, 1:, :], name=pre+"Protein_grid_%d"%z)(model_p)
            model_ps.append(model_phots)

        # Residual connection
        model_ph_residual = model_p_interaction
        model_ph_residual = self.time_distributed_dense_norm(hots_dimension, activation, name="HoTS_Protein_resiual",
                                                            dropout=dropout, params_dic=params_dic, norm=True)(model_ph_residual)
        model_ph_residual = MaxPool1D(protein_grid_size, padding='same', name="HoTS_Protein_residual_grid")(
            model_ph_residual)
        # From Transformer for BR prediction
        model_ph = model_ps[n_stack_hots_prediction-1]
        #model_ph = self.dense_norm(hots_dimension, activation, name="HoTS_Pharm_dense", dropout=dropout,
        #                           params_dic=params_dic, norm=True)(model_ph)

        # HoTS Prediction
        model_ph = Concatenate()([model_ph, model_ph_residual])
        model_hots_dense = model_ph
        if hots_fc_layers:
            input_layer_hots = model_hots_dense
            hots_fc_layers = return_tuple(hots_fc_layers)
            for z, hots_fc_layer in enumerate(hots_fc_layers):
                model_hots_dense = self.time_distributed_dense_norm(hots_fc_layer, activation=activation, norm=True,
                                                         name="HoTS_last_dense_%d_%d"%(hots_fc_layer,z), dropout=dropout,
                                                         params_dic=params_dic)(input_layer_hots)
                input_layer_hots = model_hots_dense

        model_hots = self.time_distributed_dense_norm((len(anchors))*3, name='HoTS_pooling_feature_last', dropout=0.0,
                                                     activation='sigmoid', params_dic=params_dic, norm=True)(model_hots_dense)
        model_ph_mask = Permute([2, 1])(RepeatVector(len(anchors)*3)(input_mask))
        model_ph_mask = Lambda(lambda a: a[:, 1:, :])(model_ph_mask)
        model_hots = Multiply()([model_hots, model_ph_mask])

        model_hots = Reshape((-1, len(anchors), 3), name='HoTS_pooling_reshape')(model_hots)

        # Protein Residual
        model_p = GlobalMaxPool1D()(model_p_interaction)
        model_p = self.dense_norm(hots_dimension, activation, name="DTI_Protein_residual",
                                  dropout=dropout, params_dic=params_dic, norm=True)(model_p)
        # Drug Residual
        model_d = model_ds[-1] #self.dense_norm(hots_dimension, activation, name="DTI_Drug_dense", dropout=dropout,
                               #    params_dic=params_dic, norm=True)(model_ds[-1])

        # Transformer for DTI prediction
        model_pd = model_pds[-1]#self.dense_norm(hots_dimension, activation, name="DTI_Pharm_dense", dropout=dropout,
                   #                params_dic=params_dic, norm=True)(model_pds[-1])#BatchNormalization()(model_pds[-1])
        #model_t = Concatenate()([model_p, model_d])
        model_t = Concatenate()([model_pd, model_p, model_d])#Concatenate()([model_pd, model_p, model_d]) # Ablation No Attention : Concatenate()([model_p, model_d]) # Ablation No Residual: Concatenate()([model_d, model_p])

        # DTI prediction
        input_t = model_t
        if fc_layers is not None:
            fc_layers = return_tuple(fc_layers)
            for z, fc_layer in enumerate(fc_layers):
                model_t = self.dense_norm(fc_layer, activation, dropout=dropout, norm=True,
                                                name="FC_%d"%fc_layer, params_dic=params_dic)(input_t)
                input_t = model_t
        model_t = Dense(1, name="DTI_prediction")(model_t)
        model_t = Activation('sigmoid')(model_t)

        # Definining models
        self.model_hots = Model(inputs=[input_d, input_p, input_mask], outputs=model_hots)
        self.model_t = Model(inputs=[input_d, input_p, input_mask], outputs = model_t)

    '''
    def gelu(self, x):
        return 0.5 * x * (1 + K.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * K.pow(x, 3))))
    '''

    def PLayer(self, size, filters, activation, dropout, params_dic, norm=True, name=""):
        def f(input):
            if norm:
                model_conv = Convolution1D(filters=filters, kernel_size=size, padding='same', name=name, **params_dic)(input)
                #model_conv = WeightNormalization(model_conv, name=name+"_norm")(input)
                #model_conv = BatchNormalization(name=name+"_meanonly_batchnorm", scale=False)(model_conv)
                model_conv = LayerNormalization(name=name + "_layernorm")(model_conv)
            else:
                model_conv = Convolution1D(filters=filters, kernel_size=size, padding='same', name=name, **params_dic)(input)
            model_conv = Activation(activation, name=name+"_activation")(model_conv)
            '''
            if norm:
                model_conv = LayerNormalization(name=name + "_norm")(model_conv)
            '''
            model_conv = Dropout(dropout, name=name+"_dropout")(model_conv)
            return model_conv
        return f

    def repeat_vector(self, args):
        layer_to_repeat = args[0]
        sequence_layer = args[1]
        to_be_repeated = K.expand_dims(layer_to_repeat,axis=1)
        one_matrix = K.ones_like(sequence_layer[:,:1])
        return K.batch_dot(one_matrix,to_be_repeated)

    def dense_norm(self, units, activation, name=None, dropout=0.0, params_dic=None, norm=True):
        def dense_func(input):
            if norm:
                densed = Dense(units, name=name, **params_dic)(input)
                densed = BatchNormalization(name=name+"_batchnorm")(densed)
            else:
                densed = Dense(units, name=name, **params_dic)(input)
            densed = Activation(activation, name=name+"_activation")(densed)
            densed = Dropout(dropout, name=name+"_dropout")(densed)
            return densed
        return dense_func

    def time_distributed_dense_norm(self, units, activation, name=None, dropout=0.0, params_dic=None, norm=True, use_bias=True):
        def dense_func(input):
            if norm:
                densed = LayerNormalization(name=name + "_norm")(input)
                densed = Dense(units, name=name, use_bias=use_bias, **params_dic)(densed)
            else:
                densed = Dense(units, name=name, use_bias=use_bias, **params_dic)(input)
            densed = Activation(activation, name=name+"_activation")(densed)
            densed = Dropout(dropout, name=name+"_dropout")(densed)
            return densed
        return dense_func

    def attention(self, n_dims=32, dropout=0.2, name=None):
        def attn_func(query, key, value, attention_mask=None):
            dot = Dot([2,2], name=name+"_dot")([query, key])
            dot = Lambda(lambda a: a/tf.sqrt(tf.cast(K.shape(a)[-1], tf.float32)), name=name+"_scaled_dot")(dot)
            if attention_mask is not None:
                dot = Multiply()([dot, attention_mask])
                dot = Lambda(lambda a: tf.where(a!=0, a, -1e9*tf.ones_like(a)))(dot)
            attn = Activation('softmax', name=name+"_softmax")(dot)
            attn = Dropout(dropout)(attn)
            attn = Dot([2,1], name=name+"_attended")([attn, value])
            return attn
        return attn_func

    def self_attention(self, n_dims, activation, dropout, params_dic, name="", n_heads=4):
        def self_attn_func(layer_q, layer_k, layer_v, attention_mask=None):
            # Queries
            _, _, model_partial_q_dim = K.int_shape(layer_q)
            model_partial_q_dim = int(model_partial_q_dim/n_heads)
            model_q = Dense(n_dims, activation=None, name="%s_multihead_queries"%(name))(layer_q)
            model_qs = [self.crop(i*model_partial_q_dim, (i+1)*model_partial_q_dim, name="%s_query_cropped_%d"%(name, i))(model_q) for i in range(n_heads)]
            # Keys
            _, _, model_partial_k_dim = K.int_shape(layer_k)
            model_partial_k_dim = int(model_partial_k_dim/n_heads)
            model_k = Dense(n_dims, activation=None, name="%s_multihead_keys"%(name))(layer_k)
            model_ks = [self.crop(i*model_partial_k_dim, (i+1)*model_partial_k_dim, name="%s_key_cropped_%d"%(name, i))(model_k) for i in range(n_heads)]
            # Values
            _, _, model_v_dims = K.int_shape(layer_v)
            model_partial_v_dim = int(model_v_dims/n_heads)
            model_v = Dense(n_dims, activation=None, name="%s_multihead_values"%(name))(layer_k)
            model_vs = [self.crop(i*model_partial_v_dim, (i+1)*model_partial_v_dim,
                                  name="%s_value_cropped_%d"%(name, i))(model_v) for i in range(n_heads)]
            # Self-attention
            model_attn = [self.attention(model_v_dims, name="%s_multihead_%d"%(name, i),
                                            dropout=dropout)(model_kqvs[0], model_kqvs[1], model_kqvs[2], attention_mask)
                             for i, model_kqvs in enumerate(zip(model_qs, model_ks, model_vs))]
            if len(model_attn)==1:
                model_attn = model_attn[0]
            else:
                model_attn = Concatenate()(model_attn)
            # W_model
            model_attn = self.time_distributed_dense_norm(n_dims, None, dropout=dropout, norm=False,
                                                       name="%s_self_attention_dense"%name, params_dic=params_dic)(model_attn)
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

    def get_model_hots(self):
        return self.model_hots

    def get_model_t(self):
        return self.model_t
