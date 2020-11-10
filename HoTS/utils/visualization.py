import numpy as np
import re


def split_string(string, line_length=100, prefix="           "):
    return ["{:>10} : ".format(prefix)+split for split in re.findall(".{1,%d}"%line_length, string)]


def print_binding(seq, prediction, line_length=100, ground_truth=None, output_file=None, print_score=True):
    seq_splitted = split_string(seq, line_length=line_length, prefix='Sequence')
    max_len = len(seq)
    pred_binding = ' '*len(seq)
    pred_score = ' '*len(seq)
    pred_binding = list(pred_binding)
    pred_score = list(pred_score)
    for pred_start, pred_end, score in prediction:
        for i in range(pred_start, pred_end):
            if i < max_len:
                pred_binding[i] = seq[i]
        ## print score
        if pred_start+2 < max_len:
            pred_score[pred_start] = str(score*100)[0]
            pred_score[pred_start+1] = str(score*100)[1]
            pred_score[pred_start+2] = "%"
    pred_binding = "".join(pred_binding)
    pred_score = "".join(pred_score)
    pred_binding_splitted = split_string(pred_binding, line_length=line_length, prefix="Prediction")
    pred_score_splitted = split_string(pred_score, line_length=line_length, prefix='Score')
    if ground_truth:
        true_binding = ' '*len(seq)
        true_binding = list(true_binding)
        for true_index in ground_truth:
            true_start, true_end = true_index
            for i in range(true_start, true_end):
                if i < max_len:
                    true_binding[i] = seq[i]
        true_binding = "".join(true_binding)
        true_binding_splitted = split_string(true_binding, line_length=line_length, prefix="True")
        for seq, true, pred, score in zip(seq_splitted, true_binding_splitted, pred_binding_splitted, pred_score_splitted):
            print(seq)
            print(true)
            print(pred)
            if print_score:
                print(score)
            print(" "*len(seq))
            if output_file:
                output_file.write('\n'.join([seq, true, pred])+'\n')
    else:
        for seq, pred, score in zip(seq_splitted, pred_binding_splitted, pred_score_splitted):
            print(seq)
            print(pred)
            if print_score:
                print(score)
            if output_file:
                output_file.write('\n'.join([seq, pred])+'\n')