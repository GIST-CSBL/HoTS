import numpy as np
import pandas as pd
from HoTS.utils.metric import IoU
from keras.preprocessing.sequence import pad_sequences
from rdkit import Chem
from rdkit.Chem import AllChem
from ast import literal_eval
from tqdm import tqdm

aa = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
seq_dic = {w: i+1 for i,w in enumerate(aa)}
seq_r_dic = {i+1:w for i,w in enumerate(aa)}
seq_r_dic.update({0:" "})


secondary_structures = ['A', 'B', 'C', 'T']
secondary_structure_dic = {c: i+1 for i, c in enumerate(secondary_structures)}

atoms = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
        "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
        "9": 39, "8": 7,
        "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
        "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
        "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
        "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
        "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
        "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
atom_dic =  {atom: i+1 for i, atom in enumerate(atoms)}

class ProteinEncoder(object):
    def __init__(self, feature="Sequence", **kwargs):
        self.__feature = feature
        self.__pv = None
        self.__pssm_dict = None

    def get_ProtVec_feature(self, protein):
        feature = []
        if len(protein)==0:
            return np.zeros(shape=(5,100))
        for i in range(len(protein)-2):
            try:
                mer_3 = protein[i:i+3]
                feature.append(self.__pv[mer_3])
            except:
                mer_e = np.zeros(shape=(100))
                feature.append(mer_e)
        return np.stack(feature)

    def get_type(self):
        return self.__feature

    def pad_2d_feature(self, features, max_len):
        #max_len = max([len(feature) for feature in features])
        padded_features = []
        for feature in features:
            if max_len==0:
                n_pad = 10
            else:
                n_pad = max_len - feature.shape[0]
            padded_features.append(np.concatenate([feature,np.zeros(shape=(n_pad, feature.shape[1]))]))
        return np.stack(padded_features)

    def load_pssm(self, pssm_json=None):
        import json
        f = open(pssm_json)
        pssm_dict = json.load(f)
        f.close()
        self.__pssm_dict = pssm_dict

    def load_prot_vec(self, pv=None):
        self.__pv = pv

    def encode(self, sequence):
        if self.__feature == "Sequence":
            encoded_sequence = [seq_dic[aa] for aa in sequence]
        elif self.__feature == "ProtVec":
            encoded_sequence = self.get_ProtVec_feature(sequence)
        return encoded_sequence

    def encode_pssm(self, protein_id):
        encoded_sequence = self.__pssm_dict[protein_id]
        return encoded_sequence

    def pad(self, sequences, max_len=None):
        if not max_len:
            max_len = max([len(sequence) for sequence in sequences])
        if not self.__pv:
            padded_sequence = pad_sequences(sequences, padding='post', maxlen=max_len)
        else:
            padded_sequence = self.pad_2d_feature(sequences, max_len)
        return padded_sequence

    def pad_pssm(self, sequences, max_len):
        padded_sequence = self.pad_2d_feature(sequences, max_len)
        return padded_sequence


class CompoundEncoder(object):

    def __init__(self, feature="Convolution", radius=2, n_bits=2048, **kwargs):
        self.__feature = feature
        self.__radius = radius
        self.__n_bits = n_bits
        if feature=="SMILES_Sentence":
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()

    def load_sentence_model(self, model_dir):
        self.sp.Load(model_dir)

    def get_type(self):
        return self.__feature

    def mol_to_morgan(self, mol):
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, self.__radius, nBits=self.__n_bits))
        return fp

    def encode(self, SMILES):
        if self.__feature=='Morgan':
            mol = Chem.MolFromSmiles(SMILES)
            morgan_feature = self.mol_to_morgan(mol)
            return morgan_feature
        elif self.__feature=="SMILES":
            smiles_feature = [atom_dic[atom] for atom in SMILES if atom in atom_dic.keys()]
            smiles_feature = list(filter(None, smiles_feature))
            #print(sentence_feature)
            return smiles_feature
        elif self.__feature=="SMILES_Sentence":
            sentence_feature = self.sp.EncodeAsPieces(''.join([i for i in SMILES if not i.isdigit()]))
            sentence_feature = [self.sp.PieceToId(i) for i in sentence_feature]
            return sentence_feature

    def pad(self, compounds, max_len):
        return pad_sequences(compounds, padding='post', maxlen=max_len)

def permute_data(embeddings, adj_mats,max_length=200, seed=None):
    if seed:
        np.random.seed(seed)
    permuted_idx = np.random.permutation(range(max_length))
    result_embeddings = embeddings[:, permuted_idx]
    result_adj_mats = adj_mats.copy()
    result_adj_mats[:,permuted_idx,:] = result_adj_mats[:,:,permuted_idx]
    result_adj_mats[:,:,permuted_idx] = result_adj_mats[:,permuted_idx,:]
    return result_embeddings, result_adj_mats


def parse_DTI_data(dti_dir, drug_dir, protein_dir, with_label=True, prot_len=2500, prot_vec=None,
                   drug_vec=None, drug_len=2048, protein_encoder=None, compound_encoder=None, return_dti=False,
                   pssm_json=None, return_sequence=False):

    print("Parsing {0} , {1}, {2} with length {3}, type {4}".format(*[dti_dir ,drug_dir, protein_dir, prot_len, prot_vec]))

    protein_col = "Protein_ID"
    drug_col = "Compound_ID"
    col_names = [protein_col, drug_col]
    if with_label:
        label_col = "Label"
        col_names += [label_col]
    dti_df = pd.read_csv(dti_dir)
    drug_df = pd.read_csv(drug_dir, index_col="Compound_ID")
    protein_df = pd.read_csv(protein_dir, index_col="Protein_ID")
    tqdm.pandas()
    #smiles = drug_df.SMILES.tolist()
    if compound_encoder is not None:
        drug_vec=compound_encoder.get_type()
        print("Encoding compound with %s type"%drug_vec)
        #encoded_drug = [compound_encoder.encode(s) for s in smiles]
        drug_df["drug_feature"] = drug_df.SMILES.progress_map(compound_encoder.encode)
        print("Encoding compound ends!")

    else:
        drug_df["drug_feature"] = drug_df[drug_vec].map(lambda fp: [int(bit) for bit in fp.split("\t")])
    dti_df = pd.merge(dti_df, protein_df, left_on=protein_col, right_index=True, how='left')
    dti_df = pd.merge(dti_df, drug_df, left_on=drug_col, right_index=True, how='left')
    #print(dti_df[[protein_col, drug_col, "Sequence"]])
    if prot_vec=="PSSM":
        protein_feature = dti_df[protein_col].map(protein_encoder.encode).tolist()
    else:
        protein_feature = dti_df["Sequence"].map(protein_encoder.encode).tolist()
    #print(protein_feature[0:32])
    result_dic = {"protein_feature": protein_feature}
    if with_label:
        label = dti_df[label_col].values
        print("\tPositive data : %d" %(sum(dti_df[label_col])))
        print("\tNegative data : %d" %(dti_df.shape[0] - sum(dti_df[label_col])))
        result_dic.update({"label": label})
    if drug_vec.split("_")[0]=="SMILES":
        drug_feature = dti_df["SMILES"].map(compound_encoder.encode).tolist()
    else:
        drug_feature = np.stack(dti_df["drug_feature"])
    result_dic.update({"drug_feature": drug_feature})
    if return_dti:
        result_dic.update({"Compound_ID":dti_df.Compound_ID.tolist(), "Protein_ID":dti_df.Protein_ID.tolist()})
    if return_sequence:
        result_dic.update({"sequence":dti_df["Sequence"]})
    return result_dic

def get_anchor_index(binding_length, anchors):
    #permuted_index = np.random.permutation(len(anchors))
    permuted_index = range(len(anchors))
    for i in permuted_index:
        anchor = anchors[i]
        if (binding_length < (anchor*np.e)) & (binding_length > anchor):
            return i, anchor
            #result_index.append(i)
            #result_anchors.append(anchor)
    if binding_length < anchors[0]:
        return 0, anchors[0]
    else:
        return len(anchors)-1, anchors[-1]

def parse_HoTS_data(path_to_hots, name_index="Protein_ID", pdb_bound=False, compound_encoder=None, protein_encoder=None,
                    binding_region=True):
    print("Parsing HoTS data: %s"%path_to_hots)
    hots_df = pd.read_csv(path_to_hots, sep='\t')
    print("Number of 3D-complexes : %d" % hots_df.shape[0])
    print("Number of proteins : %d" % np.unique(hots_df[name_index]).shape[0])
    tqdm.pandas()
    if protein_encoder.get_type()=="PSSM":
        seq = hots_df.Protein_ID.map(protein_encoder.encode).tolist()
    else:
        seq = hots_df.Sequence.progress_map(protein_encoder.encode).tolist()
    smiles = hots_df.SMILES.tolist()
    original_seqs = hots_df.Sequence.tolist()
    if compound_encoder.get_type().split("_")[0]=="SMILES":
        ligand = [compound_encoder.encode(s) for s in smiles]
    else:
            ligand = np.stack([compound_encoder.encode(s) for s in smiles])
    names = hots_df[name_index].tolist()
    result_dic = {"protein_feature":seq, "drug_feature": ligand,
                  "protein_names":names, "sequence":original_seqs}
    if binding_region:
        hots_df["binding_region"] = hots_df.binding_region.map(literal_eval)
        hots_df["binding_index"] = hots_df.binding_index.map(literal_eval)
        sites = hots_df.binding_index.tolist()
        indice = hots_df.binding_region.tolist()
        result_dic.update({"site_feature":sites, "index_feature":indice})
    if pdb_bound:
        pdb_starts = hots_df["PDB_start"].tolist()
        pdb_ends = hots_df["PDB_end"].tolist()
        result_dic.update({"pdb_starts":pdb_starts, "pdb_ends":pdb_ends})
    return result_dic

def round_value(value, maximum_value):
    if value > maximum_value:
        return maximum_value
    elif value - 1 < 0:
        return 0
    else:
        return int(np.round(value))

def get_grid_index(start, end, grid_size):
    start_index = int(np.floor(start/grid_size))
    end_index = int(np.ceil(end/grid_size))
    return list(range(start_index, end_index))


def create_HoTS_output(binding_sites, sample_index, anchors, max_len=100, grid_len=25, train=True):
    n_grids = int(np.ceil(max_len/grid_len))
    n_anchors = len(anchors)
    hots_grids = np.zeros(shape=(n_grids, n_anchors, 3), dtype=np.float32)
    indexing_result = []
    for binding_start, binding_end in binding_sites:
        binding_median = int((binding_start + binding_end)/2)
        binding_len = (binding_end - binding_start)
        # Generate Positives
        # Generate Positives for small binding sites
        if binding_len < anchors[0]:
            grid_index, binding_remainder = np.divmod(binding_median, grid_len)
            anchor = anchors[0]
            c_p = binding_remainder/grid_len
            c_t = binding_median
            w_p = 0.0
            w_t = int(np.round(anchor * np.exp(w_p)))
            hots_grids[int(grid_index), 0, :] = [c_p, w_p, 1]
            indexing_result.append((sample_index, c_t, w_t, 0, 1))
        elif binding_len >= int(anchors[-1] * np.e):
            grids = get_grid_index(binding_start, binding_end, grid_len)
            for grid_index in grids:
                c_p = np.random.uniform(0, 1)
                c_t = c_p*grid_len + grid_index*grid_len
                anchor = anchors[-1]
                w_p = np.random.uniform(0, 1)
                w_t = int(np.round(anchor*np.exp(w_p)))
                s_t = round_value(c_t - w_t/2, max_len)
                e_t = round_value(c_t + w_t/2, max_len)
                is_in_br = [(s >= binding_start) & (s <= binding_end) for s in range(s_t, e_t)]
                if sum(is_in_br)/len(is_in_br) >= 0.7:
                    hots_grids[int(grid_index), len(anchors)-1, :] = [c_p, w_p, 1]
                    indexing_result.append((sample_index, c_t, w_t, len(anchors)-1, 1))
        # Generate positives which is longer than anchor
        else:
            grid_index, binding_remainder = np.divmod(binding_median, grid_len)
            trial = 0
            while True:
                wps = []
                wts = []
                ious = []
                c_p = binding_remainder/grid_len
                c_t = binding_median
                for anchor in anchors:
                    w_p = np.random.uniform(0, 1)
                    w_t = int(np.round(anchor*np.exp(w_p)))
                    s_t = round_value(c_t - w_t/2, max_len)
                    e_t = round_value(c_t + w_t/2, max_len)
                    wps.append(w_p)
                    wts.append(w_t)
                    ious.append(IoU(binding_start, binding_end, s_t,  e_t, mode='se'))
                if np.any(np.array(ious)>=0.7):
                    index = np.argmax(ious)
                    hots_grids[int(grid_index), index, :] = [c_p, wps[index], 1]
                    indexing_result.append((sample_index, c_t, wts[index], anchors[index], 1))
                    break
                trial += 1
                if trial > 100:
                    break
    '''
    n_neg = 0

    # Sampling negatives IoU with true binding region < 0.2
    neg_trial = 0
    while n_neg < 10*len(indexing_result):
        grid_selected = np.random.randint(n_grids)
        c_p = np.random.randint(0, grid_len)
        c_t = c_p + grid_selected*grid_len
        w_p = np.random.random()
        anchor_index = np.random.randint(len(anchors))
        anchor_selected = anchors[anchor_index]
        w_t = np.round(anchor_selected * np.exp(w_p))
        s_t = round_value(c_t - w_t/2, max_len)
        e_t = round_value(c_t + w_t/2, max_len)
        if np.all([IoU(binding_start, binding_end, s_t, e_t, mode='se') <= 0.2 for binding_start, binding_end in binding_sites]):
            hots_grids[int(grid_selected), anchor_index, :] = [(c_p/grid_len)*2 - 1, w_p*2 - 1, -1]
            n_neg += 1
        neg_trial += 1
        if neg_trial >= 300:
            break
    '''

    return hots_grids, indexing_result
