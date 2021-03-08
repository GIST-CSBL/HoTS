import numpy as np
from multiprocessing import Pool

class HoTSPooling(object):
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