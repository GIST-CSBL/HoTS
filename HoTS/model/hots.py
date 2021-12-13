import numpy as np
from keras.utils import Progbar
from keras import models
from HoTS.model.model import HoTSModel
from HoTS.model.loss import HoTSLoss
from HoTS.utils import *

from sklearn.metrics import precision_recall_curve, auc, roc_curve, confusion_matrix

from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.preprocessing import sequence



class HoTS(object):


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
        print("HoTS model initialization done!")

    def build_model(self, dropout=0.1, drug_layers=(1024,512), protein_strides = (10,15,20,25), filters=64,
                 hots_dimension=64, n_stack_hots_prediction=0,
                 fc_layers=None, hots_fc_layers=None, activation="relu",
                 protein_layers=None, protein_encoder_config={}, compound_encoder_config={},
                 protein_grid_size=10, anchors=[10, 15, 20], hots_n_heads=4, **kwargs):
        self.protein_grid_size = protein_grid_size
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
        self.prot_vec = protein_encoder_config["feature"]
        self.drug_vec = compound_encoder_config["feature"]
        self.drug_len = compound_encoder_config["n_bits"]
        self.activation = activation
        self.protein_layers = protein_layers
        self.protein_encoder_config = protein_encoder_config
        self.protein_encoder = ProteinEncoder(**protein_encoder_config)
        self.compound_encoder_config = compound_encoder_config
        self.compound_encoder = CompoundEncoder(**compound_encoder_config)
        hots_model = HoTSModel(self.drug_layers, self.protein_strides,self.filters, self.fc_layers, self.hots_fc_layers,
                               dropout=self.dropout, hots_dimension=self.hots_dimension,
                               activation=self.activation, protein_grid_size=self.protein_grid_size,
                                protein_layers=self.protein_layers,drug_vec=self.drug_vec, drug_len=self.drug_len, anchors=self.anchors,
                               hots_n_heads=self.hots_n_heads, n_stack_hots_prediction=self.n_stack_hots_prediction)
        self.model_hots, self.model_t = hots_model.get_model_hots(), hots_model.get_model_t()
        print("Model hyperparameters")
        for key, value in self.__dict__.items():
            print("\t%s:"%key, value)
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
            self.build_model(**class_dict)
            #self.summary()
            hots_file = class_dict["hots_file"]
            self.model_hots.load_weights(hots_file)
            #self.model_hots = models.load_model(hots_file)
            print("\tHoTS Model loaded at %s"%hots_file)
            dti_file = class_dict["dti_file"]
            self.model_t.load_weights(dti_file)
            #self.model_t = models.load_model(dti_file)
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
                      min_value=self.anchors[0], max_value=int(self.anchors[-1]*np.e)).get_AP()
        '''
        max_len = int(np.ceil(len(protein_feature[0])/20)*20)
        protein_feature_0 = sequence.pad_sequences([protein_feature[0]], maxlen=max_len, padding='post')
        drug_feature_0 = np.stack([drug_feature[0]])
        mask = np.ones(shape=(1, int(np.ceil(max_len / self.protein_grid_size)) + 1))
        #mask[0, 0: int(np.ceil(len(protein) / self.protein_grid_size)) + 1] = 1
        attention_layer_0 = Model(inputs=self.model_hots.inputs, outputs=self.model_hots.get_layer(name="DTI_Protein_feature_attention_Input").output)
        #attention_layer_1 = Model(inputs=self.model_hots.inputs, outputs=self.model_hots.get_layer(name="Protein_attention_128_0_multihead_1_softmax").output)
        #attention_layer_2 = Model(inputs=self.model_hots.inputs, outputs=self.model_hots.get_layer(name="Protein_attention_128_0_multihead_2_softmax").output)
        #attention_layer_3 = Model(inputs=self.model_hots.inputs, outputs=self.model_hots.get_layer(name="Protein_attention_128_0_multihead_3_softmax").output)
        print(attention_layer_0.predict([drug_feature_0,protein_feature_0, mask]))
        #mean_ap=0.0
        #print(attention_layer_1.predict([drug_feature_0,protein_feature_0]))
        #print(attention_layer_2.predict([drug_feature_0,protein_feature_0]))
        #print(attention_layer_3.predict([drug_feature_0,protein_feature_0]))
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
            train_seq_batch, train_HoTs, train_mask, train_indice, train_ligand_batch, train_dtis, train_names = train_gen.next()
            loss = self.model_hots.train_on_batch([train_ligand_batch, train_seq_batch, train_mask], train_HoTs)
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


    def training(self, dti_dataset, hots_dataset,n_epoch=10, batch_size=32, learning_rate=1e-3, decay=1e-3,
                 retina_loss_weight=2, reg_loss_weight=1., conf_loss_weight=10, negative_loss_weight=0.1,
                   hots_training_ratio=1, hots_warm_up_epoch=20, **kwargs):
        self.learning_rate = learning_rate
        self.decay = decay
        #self.opt_hots = AdamWeightDecay(lr=self.learning_rate, decay=self.decay)
        #self.opt_dti = AdamWeightDecay(lr=self.learning_rate, decay=self.decay)

        self.reg_loss_weight = reg_loss_weight
        self.conf_loss_weight = conf_loss_weight
        self.negative_loss_weight = negative_loss_weight
        self.retina_loss_weight = retina_loss_weight

        train_hots_n_steps = int(np.ceil(len(hots_dataset["index_feature"]) / batch_size)) * hots_warm_up_epoch
        train_hots_warmup_steps = train_hots_n_steps * 0.2

        train_dti_n_steps = int(np.ceil(len(dti_dataset["label"]) / batch_size)) * n_epoch
        train_dti_warmup_steps = train_dti_n_steps * 0.2

        self.hots_loss = HoTSLoss(grid_size=self.protein_grid_size, anchors=self.anchors,
                                    reg_loss_weight=self.reg_loss_weight, conf_loss_weight=self.conf_loss_weight,
                                    negative_loss_weight=self.negative_loss_weight, retina_loss_weight=self.retina_loss_weight)

        self.opt_hots = Adam(lr=learning_rate, decay=self.decay)
        self.opt_dti = Adam(lr=learning_rate, decay=self.decay)
        #self.opt_hots = create_optimizer(self.learning_rate, train_hots_n_steps, train_hots_warmup_steps)
        #self.opt_hots = create_optimizer(self.learning_rate, train_dti_n_steps, train_dti_warmup_steps)

        self.model_t.compile(optimizer=self.opt_dti, loss='binary_crossentropy', metrics=['accuracy'])
        self.model_hots.compile(optimizer=self.opt_hots, loss=self.hots_loss.compute_hots_loss)

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


    def HoTS_prediction(self, protein_feature, drug_feature, th=0., batch_size=32, **kwargs):
        test_n_steps = int(np.ceil(len(protein_feature)/batch_size))
        test_gen = DataGeneratorHoTS(protein_feature, ind_label=None, ligand=drug_feature, name=None, anchors=self.anchors,
                                     batch_size=batch_size, train=False, shuffle=False, compound_encoder=self.compound_encoder,
                                     protein_encoder=self.protein_encoder, grid_size=self.protein_grid_size)
        predicted_dtis = []
        predicted_indice = []
        for j in range(test_n_steps):
            test_seq, test_mask, test_ligand = test_gen.next()
            test_max_len = test_seq.shape[1]
            prediction_dti = self.model_t.predict_on_batch([test_ligand, test_seq, test_mask])
            prediction_hots = self.model_hots.predict([test_ligand, test_seq, test_mask])
            hots_pooling = HoTSPooling(self.protein_grid_size, max_len=test_max_len, anchors=self.anchors,
                                        protein_encoder=self.protein_encoder)
            predicted_pooling_index = hots_pooling.hots_grid_to_subsequence(test_seq, prediction_hots, th=th)
            predicted_dtis.append(prediction_dti)
            predicted_indice += predicted_pooling_index
        predicted_dtis = np.concatenate(predicted_dtis)
        return predicted_dtis, predicted_indice


