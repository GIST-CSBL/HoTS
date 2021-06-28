import numpy as np
from sklearn.metrics import auc
from multiprocessing import Pool
from tqdm import tqdm, tqdm_notebook


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
        self.true_called = np.zeros(self.n_pred)

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
        #precisions = []
        #recalls = []
        pool = Pool(50)
        pred_ind_for_gt = [(pred_start, pred_end, self.true_inds[sample_ind], self.true_inds_called[sample_ind])
                           for sample_ind, pred_start, pred_end, score in self.pred_inds]
        call_results = pool.starmap(self.call_prediction_gt, pred_ind_for_gt)
        pool.close()
        pool.terminate()
        pool.join()
        precisions = np.cumsum([pred_called for pred_called, true_inds_called, is_true_called in call_results])
        recalls = np.cumsum([is_true_called for pred_called, true_inds_called, is_true_called in call_results])
        '''
        for pred_ind, call_result in tqdm(zip(self.pred_inds, call_results), desc="PR_calculation", total=len(call_results),
                                          bar_format="{l_bar}{r_bar}"):
            sample_ind, pred_start, pred_end, score = pred_ind
            pred_called, true_inds_called = call_result#self.call_prediction_gt(pred_start, pred_end, self.true_inds[sample_ind],
                                            #                        self.true_inds_called[sample_ind], self.min_value, self.max_value)
            pred_calls.append(pred_called)
            precisions.append(sum(pred_calls))
            #self.true_inds_called[sample_ind] = true_inds_called
            recalls.append(self.get_n_called_true())
        '''
        self.precisions = np.array(precisions)/range(1, len(call_results)+1)
        #self.precisions = np.triu(np.tile(self.precisions, (len(self.precisions), 1)), 0).max(axis=1)
        self.recalls = np.array(recalls)/self.n_true
        #for k in range(len(self.precisions) - 2):
        #    self.precisions[k] = max(self.precisions[k + 1:])
        interpolated_precisions = []
        interpolated_precisions.append(np.max(self.precisions))
        for i in range(1, 10):
            recall_ind = np.argmin(np.abs(self.recalls - (i/10)))
            precision = np.max(self.precisions[recall_ind:])
            interpolated_precisions.append(precision)
        #interpolated_precisions.append(self.precisions[-1])
        #print(interpolated_precisions)
        #ap = np.mean(interpolated_precisions)  # auc(recalls, precisions)
        ap = auc(self.recalls, self.precisions)
        return ap


    def call_prediction_gt(self, pred_start, pred_end, true_indices, true_called, th=0.5):

        pred_call = False
        true_called = true_called.copy()
        is_true_called = False
        for i, true_start_end in enumerate(true_indices):
            true_start, true_end = true_start_end

            if (true_end - true_start) <= self.min_value:
                if true_called[i]==True:
                    continue
                true_called_with_pred = [(pred_start <= s) & (pred_end >= s) for s in range(true_start, true_end)]
                if sum(true_called_with_pred)/len(true_called_with_pred)>=th:
                    pred_call = True
                    true_called[i] = True
                    is_true_called = True
            if (true_end - true_start) > self.max_value :
                if true_called[i] == True:
                    is_true_called = True
                    continue
                prediction_called = [(true_start <= s) & (true_end >= s)  for s in range(pred_start, pred_end)]
                if (sum(prediction_called)/len(prediction_called)) >= th:
                    pred_call = True
                true_called_with_pred = [(pred_start <= s) & (pred_end >= s) for s in range(true_start, true_end)]
                true_called[i] = [(s_1 | s_2) for s_1, s_2 in zip(true_called_with_pred, true_called[i])]
                if sum(true_called[i])/len(true_called[i]) >= th:
                    true_called[i] = True
                    is_true_called = True
            else:
                if true_called[i] == True:
                    continue
                if IoU(true_start, true_end, pred_start, pred_end, mode='se') >= th:
                    pred_call = True
                    true_called[i] = True
                    is_true_called = True
        return pred_call, true_called, is_true_called




