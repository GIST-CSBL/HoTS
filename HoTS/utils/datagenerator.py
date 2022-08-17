import numpy as np
from tensorflow.keras.preprocessing import sequence
from .build_features import create_HoTS_output


class DataGeneratorDTI(object):
    """

    """
    def __init__(self, train_drug, train_protein, train_label=None, batch_size=32,
                 protein_encoder=None, compound_encoder=None, train=True, grid_size=10, protein_max_len=2500):
        self.__batch_size = batch_size
        self.protein_encoder = protein_encoder
        self.compound_encoder = compound_encoder
        self.train = train
        self.n = len(train_drug)
        self.__grid_size = grid_size
        self.protein_max_len = protein_max_len
        if train:
            permute_index = np.random.permutation(range(self.n))
        else:
            permute_index = range(self.n)
        self.__drugs = [train_drug[i] for i in permute_index]
        self.__compound_type = compound_encoder.get_type()
        self.__proteins = [train_protein[i] for i in permute_index]
        if self.train:
            self.__label = [train_label[i] for i in permute_index]
        else:
            self.__label = None
        self.num = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.num+1)*self.__batch_size >= self.n:
            start_ind = self.num*self.__batch_size
            end_ind = self.n
        elif self.num*self.__batch_size <= self.n:
            start_ind = self.num*self.__batch_size
            end_ind = (self.num+1)*self.__batch_size
        else:
            raise StopIteration()
        self.num += 1
        drugs = self.__drugs[start_ind:end_ind]
        proteins = self.__proteins[start_ind:end_ind]
        max_len = max([len(seq) for seq in proteins])
        units = self.__grid_size
        max_len = min(int(np.ceil(max_len/units)*units), self.protein_max_len)
        protein_features = self.protein_encoder.pad(proteins, max_len=max_len)
        mask = np.zeros(shape=(end_ind - start_ind, int(np.ceil(max_len / self.__grid_size)) + 1))
        for i, protein in enumerate(proteins):
            mask[i, 0: int(np.ceil(len(protein) / self.__grid_size)) + 1] = 1
        if self.__compound_type.split("_")[0]=="SMILES":
            smiles_max_len = max([len(smiles) for smiles in drugs])
            smiles_max_len = int(np.ceil(smiles_max_len/units)*units)
            drugs = self.compound_encoder.pad(drugs, smiles_max_len)
            if self.train:
                labels = self.__label[start_ind: end_ind]
                labels = np.array(labels)
                return [drugs, protein_features], labels
            else:
                return [drugs, protein_features]
        else:
            drugs = np.stack(drugs)
            if self.train:
                labels = self.__label[start_ind: end_ind]
                labels = np.array(labels)
                return [drugs, protein_features, mask], labels

            else:
                return [drugs, protein_features, mask]


class DataGeneratorHoTS(object):
    """

    """
    def __init__(self, protein, ind_label=False, ligand=False, name=False, anchors=False,
                 batch_size=32, shuffle=True, train=True, protein_encoder=None, compound_encoder=None, n_cores=10,
                 grid_size=25):
        self.__batch_size = batch_size
        self.shuffle = shuffle
        self.train = train
        self.n = len(protein)
        if self.shuffle:
            permute_index = np.random.permutation(range(self.n))
        else:
            permute_index = range(self.n)
        self.__dti_label = np.array([1]*self.n)
        '''
        if self.train:
            protein, ligand, name, ind_label, dti_label = self.sample_negative(protein, ligand, name, ind_label)

            self.__dti_label = [dti_label[i] for i in permute_index]
        '''
        if ind_label:
            self.__ind_label = [ind_label[i] for i in permute_index]
        else:
            self.__ind_label = None
        self.__ligand = [ligand[i] for i in permute_index]
        if name:
            self.__name = [name[i] for i in permute_index]
        else:
            self.__name = None
        self.__protein = [protein[i] for i in permute_index]
        self.anchors = anchors
        self.num = 0
        self.__protein_encoder = protein_encoder
        self.__compound_encoder = compound_encoder
        self.__compound_type = self.__compound_encoder.get_type()
        self.__grid_size = grid_size
    '''
    def jaccard_similarity(self, list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    def sample_negative(self, proteins, ligands, names, ind_labels):
        pos_size = len(proteins)
        train_tuples = [(protein, ligand, ind_label) for protein, ligand, ind_label in zip(proteins, ligands, ind_labels)]
        neg_proteins = []
        neg_ligands = []
        neg_names = ["%d_negative"%i for i in range(pos_size)]
        neg_inds = []
        dti_labels = [1]*pos_size + [0]*pos_size
        for protein, ligand, ind_label in train_tuples:
            rand_ind = np.random.randint(0, pos_size)
            neg_ligand_selected = ligands[rand_ind]
            rand_val = np.random.random_sample()
            if rand_val > (1-self.jaccard_similarity(ligand, neg_ligand_selected)):
                n_inds = np.ceil(len(ind_label)*rand_val)
                rand_inds = [ind_label[i] for i in np.random.randint(0, len(ind_label), size=int(n_inds))]
                neg_inds.append(rand_inds)
            else:
                neg_inds.append([])
            neg_proteins.append(protein)
            neg_ligands.append(neg_ligand_selected)
        sampled_proteins = proteins + neg_proteins
        neg_ligands = np.stack(neg_ligands)
        sampled_ligands = np.concatenate([ligands, neg_ligands], axis=0)
        if names:
            sampled_names = names + neg_names
        else:
            sampled_names = names
        sampled_ind_label = ind_labels + neg_inds
        return sampled_proteins, sampled_ligands, sampled_names, sampled_ind_label, dti_labels
    '''

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.num+1)*self.__batch_size >= self.n:
            start_ind = self.num*self.__batch_size
            end_ind = self.n
        elif self.num*self.__batch_size <= self.n:
            start_ind = self.num*self.__batch_size
            end_ind = (self.num+1)*self.__batch_size
        else:
            raise StopIteration()
        self.num += 1
        sequences = self.__protein[start_ind:end_ind]
        max_len = max([len(seq) for seq in sequences])
        units = self.__grid_size
        max_len = int(np.ceil(max_len/units)*units)
        returning_sequence = self.__protein_encoder.pad(self.__protein[start_ind:end_ind], max_len=max_len)
        queried_ligands = self.__ligand[start_ind:end_ind]
        mask = np.zeros(shape=(end_ind - start_ind, int(np.ceil(max_len / self.__grid_size)) + 1))
        for i, protein in enumerate(sequences):
            mask[i, 0: int(np.ceil(len(protein) / self.__grid_size)) + 1] = 1
        #print(self.__compound_type)
        if self.__compound_type.split("_")[0]=="SMILES":
            smiles_max_len = max([len(smiles) for smiles in queried_ligands])
            smiles_max_len = int(np.ceil(smiles_max_len/units)*units)
            queried_ligands = self.__compound_encoder.pad(queried_ligands, smiles_max_len)
        else:
            queried_ligands = np.stack(queried_ligands)
        entry_names = None
        if self.__name:
            entry_names = self.__name[start_ind:end_ind]
        if self.__ind_label:
            queried_indice = np.array(self.__ind_label[start_ind:end_ind], dtype=object)
            queried_dtis = np.array(self.__dti_label[start_ind:end_ind])
            hots_outputs = [create_HoTS_output(indice, i, self.anchors, max_len=max_len,
                                               train=self.train, grid_len=self.__grid_size)
                            for i, indice in enumerate(queried_indice)]
            hots_grid_outputs = np.stack([hots_output[0] for hots_output in hots_outputs], axis=0)
            hots_indice = [hots_output[1] for hots_output in hots_outputs]
            return returning_sequence, hots_grid_outputs, mask, hots_indice, queried_ligands, queried_dtis, entry_names
        else:
            return returning_sequence, mask, queried_ligands


