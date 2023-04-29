import argparse
import json
import os
import time
import numpy as np
import pickle


def get_word_ali(raw_ali):
    """
    raw_ali is a string like 'start1__word1__end1 start2__word2__end2 ...'
    """
    data = []
    meta_toks = raw_ali.split()
    for meta_tok in meta_toks:
        toks = meta_tok.split('__')
        if len(toks) == 3:
            data.append([float(toks[0]), float(toks[2])])
    return data

def find_boundary_matches(gt, pred, tolerance):
    """
    gt: list of ground truth boundaries
    pred: list of predicted boundaries
    all in seconds
    """
    gt_pointer = 0
    pred_pointer = 0
    gt_len = len(gt)
    pred_len = len(pred)
    match_pred = 0
    match_gt = 0
    while gt_pointer < gt_len and pred_pointer < pred_len:
        if np.abs(gt[gt_pointer] - pred[pred_pointer]) <= tolerance:
            match_gt += 1
            match_pred += 1
            gt_pointer += 1
            pred_pointer += 1
        elif gt[gt_pointer] > pred[pred_pointer]:
            pred_pointer += 1
        else:
            gt_pointer += 1
    return match_gt, match_pred, gt_len, pred_len

print("\nI am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_json", type=str, default="/data1/scratch/coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy_with_alignments.json", help="the fn of force alignment json file")
parser.add_argument("--exp_dir", type=str, default="/scratch/cluster/pyp/exp_pyp/discovery/word_unit_discovery/disc-23/curFeats_mean_0.9_7_forceAlign")
parser.add_argument("--tolerance", type=float, default=0.02, help="tolerance of word boundary match")
parser.add_argument("--level", type=str, default="syllable", help="choose from syllable, phone, and text (which means words)")

args = parser.parse_args()

with open(args.data_json,"r") as f:
    data_json = json.load(f)
    if 'data' in data_json:
        data_json = data_json['data']
# data_json = data_json[:5000]
with open(os.path.join(args.exp_dir, "data_dict.pkl"), "rb") as f:
    data_dict = pickle.load(f)

# first run to select systematic shift, using 1/10 of the data
# following O. Räsänen, G. Doyle, and M. C. Frank, “Pre-linguistic segmentation of speech into syllable-like units,” Cognition, 2018
tolerance = args.tolerance # tolerance setting follows the paper
shifts = np.arange(-0.06, 0.031, 0.005)
shift_res = {}
# save the segmentation result when calculating metrics, just in case we want to reuse
for shift in shifts:
    match_gt_count = 0
    match_pred_count = 0
    gt_b_len = 0
    pred_b_len = 0
    for j, item in enumerate(data_json[::10]):
        wav_key = item['caption']['wav'] if 'caption' in item else item['wav']
        raw_ali = get_word_ali(item[f'{args.level}_alignment'])
        if len(raw_ali) == 0:
            continue
        gt_b = np.unique(raw_ali).tolist()
        if wav_key not in data_dict:
            print(f"missing {wav_key} in data_dict!")
        pred_b = np.unique(data_dict[wav_key]['word_boundaries']).tolist()
        pred_b = [bb + shift for bb in pred_b]
        a, b, c, d = find_boundary_matches(gt_b, pred_b, tolerance)
        
        match_gt_count += a
        match_pred_count += b
        gt_b_len += c
        pred_b_len += d
        # if j > 10:
        #     break
    b_prec = match_pred_count / pred_b_len
    b_recall = match_gt_count / gt_b_len
    b_f1 = 2*b_prec*b_recall / (b_prec+b_recall)
    b_os = b_recall / b_prec - 1.
    b_r1 = np.sqrt((1-b_recall)**2 + b_os**2)
    b_r2 = (-b_os + b_recall - 1) / np.sqrt(2)
    b_r_val = 1. - (np.abs(b_r1) + np.abs(b_r2))/2.
    shift_res[shift] = (b_prec, b_recall, b_os, b_f1, b_r_val)

# select based on max R-val
all_f1 = [val[4] for val in shift_res.values()]
ind = np.argmax(all_f1)
best_shift = list(shift_res.keys())[ind]

match_gt_count = 0
match_pred_count = 0
gt_b_len = 0
pred_b_len = 0
for j, item in enumerate(data_json):
    wav_key = item['caption']['wav'] if 'caption' in item else item['wav']
    raw_ali = get_word_ali(item[f'{args.level}_alignment'])
    if len(raw_ali) == 0:
        continue
    gt_b = np.unique(raw_ali).tolist()
    if wav_key not in data_dict:
        print(f"missing {wav_key} in data_dict!")
    pred_b = np.unique(data_dict[wav_key]['word_boundaries']).tolist()
    pred_b = [bb + best_shift for bb in pred_b]
    a, b, c, d = find_boundary_matches(gt_b, pred_b, tolerance)
    
    match_gt_count += a
    match_pred_count += b
    gt_b_len += c
    pred_b_len += d
b_prec = match_pred_count / pred_b_len
b_recall = match_gt_count / gt_b_len
b_f1 = 2*b_prec*b_recall / (b_prec+b_recall)
b_os = b_recall / b_prec - 1.
b_r1 = np.sqrt((1-b_recall)**2 + b_os**2)
b_r2 = (-b_os + b_recall - 1) / np.sqrt(2)
b_r_val = 1. - (np.abs(b_r1) + np.abs(b_r2))/2.

print("best_shift:", best_shift)
print("boundary precision:", b_prec)
print("boundary recall: ", b_recall)
print("boundary F1:", b_f1)
print("boundary over-segmentation: ", b_os)
print("boundary R value: ", b_r_val)