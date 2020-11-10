import numpy as np
from sklearn.metrics import auc
from multiprocessing import Pool


def IoU(true_c, true_w, pred_c, pred_w, mode='cw'):
    if mode == 'cw':
        true_min = true_c - true_w/2.
        true_max = true_c + true_w/2.
        pred_min = pred_c - pred_w/2.
        pred_max = pred_c + pred_w/2.
    else:
        true_min = true_c
        true_max = true_w
        pred_min = pred_c
        pred_max = pred_w

    if true_c > pred_c:
        if (true_min) > (pred_max):
            return 0
    else:
        if (true_max) < (pred_min):
            return 0

    int_max = np.min([true_max, pred_max])
    int_min = np.max([true_min, pred_min])
    union_max = np.max([true_max, pred_max])
    union_min = np.min([true_min, pred_min])
    intersect = int_max-int_min
    union = union_max - union_min
    return float(intersect)/union

class AP_calculator(object):

    def __init__(self, true_inds, pred_inds, pdb_starts, pdb_ends, min_value=5, max_value=27, **kwargs):
        # Calculate mAP
        self.min_value = min_value
        self.max_value = max_value
        self.pred_inds = [[(pred_start, pred_end, pred_score) for pred_start, pred_end, pred_score in pred_ind
                       if ((pred_end >= pdb_start) & (pred_start <= pdb_end))]
                     for pred_ind, pdb_start, pdb_end in zip(pred_inds, pdb_starts, pdb_ends)]
        self.true_inds_called = { i: [self.build_recall_index(true_start, true_end, self.min_value, self.max_value)
                                      for true_start, true_end in sample_true_inds] for i, sample_true_inds in enumerate(true_inds)}
        self.true_inds = { i: true_ind for i, true_ind in enumerate(true_inds)}
        self.pred_inds = sum([[(i, pred_start, pred_end, pred_score) for pred_start, pred_end, pred_score in sample_pred_inds]
                              for i, sample_pred_inds in enumerate(self.pred_inds)], [])
        self.pred_inds = list(reversed(list(sorted(self.pred_inds, key=lambda a: a[3]))))
        self.n_pred = len(pred_inds)
        self.n_true = len(sum(self.true_inds.values(), []))
        self.precisions = None
        self.recalls = None

    def build_recall_index(self, start, end, min_value, max_value):
        if end-start > max_value:
            return [False] * (end - start)
        else:
            return False

    def get_n_called_true(self):
        all_true_called = sum(self.true_inds_called.values(),[])
        return sum([trued_called for trued_called in all_true_called if type(trued_called) == bool])

    def get_AP(self):
        pred_calls = []
        precisions = []
        recalls = []
        for pred_ind in self.pred_inds:
            sample_ind, pred_start, pred_end, score = pred_ind
            pred_called, true_inds_called = self.call_prediction_gt(pred_start, pred_end, self.true_inds[sample_ind],
                                                                    self.true_inds_called[sample_ind], self.min_value, self.max_value)
            pred_calls.append(pred_called)
            precisions.append(sum(pred_calls)/len(pred_calls))
            self.true_inds_called[sample_ind] = true_inds_called
            recalls.append(self.get_n_called_true()/self.n_true)

        self.precisions = np.array(precisions)
        self.recalls = np.array(recalls)
        for k in range(len(self.precisions)-2):
            self.precisions[k] = max(self.precisions[k+1:])
        ap = auc(recalls, precisions)

        interpolated_precisions = []
        for i in range(11):
            recall_ind = i/10
            precisions = self.precisions[self.recalls>=recall_ind]
            if len(precisions) > 0:
                interpolated_precisions.append(np.max(precisions))
            else:
                interpolated_precisions.append(0)

        print(interpolated_precisions)
        return ap


    def call_prediction_gt(self, pred_start, pred_end, true_indices, true_called, min_value=3, max_value=27, th=0.5):
        pred_call = False
        true_called = true_called.copy()
        for i, true_start_end in enumerate(true_indices):
            true_start, true_end = true_start_end

            if (true_end - true_start) <= min_value:
                if true_called[i]==True:
                    continue
                true_called_with_pred = [(pred_start <= s) & (pred_end >= s) for s in range(true_start, true_end)]
                if sum(true_called_with_pred)/len(true_called_with_pred)>=th:
                    pred_call = True
                    true_called[i] = True
            if (true_end - true_start) > max_value :
                if true_called[i] == True:
                    continue
                prediction_called = [(true_start <= s) & (true_end >= s)  for s in range(pred_start, pred_end)]
                if (sum(prediction_called)/len(prediction_called)) >= th:
                    pred_call = True
                true_called_with_pred = [(pred_start <= s) & (pred_end >= s) for s in range(true_start, true_end)]
                true_called[i] = [(s_1 | s_2) for s_1, s_2 in zip(true_called_with_pred, true_called[i])]
                if sum(true_called[i])/len(true_called[i]) >= th:
                    true_called[i] = True
            else:
                if true_called[i] == True:
                    continue
                if IoU(true_start, true_end, pred_start, pred_end, mode='se') >= th:
                    pred_call = True
                    true_called[i] = True
        return pred_call, true_called




