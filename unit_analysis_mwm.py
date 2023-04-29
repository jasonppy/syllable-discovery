# Author: Wei-Ning Hsu, Puyuan Peng
import argparse
import code
import json
import os
import time
import torch
import numpy as np
import pickle
from collections import defaultdict, Counter

import networkx as nx
from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
from scipy.sparse import csr_matrix

STOP_WORDS = ['<blank>', '<BLANK>', '<blank>*<blank>', '<BLANK>*<BLANK>']
################################################################################
def prepare_data(centroid, json_fn, max_n_utts, exp_dir, run_length_encoding=False, A=None, b=None):
    utt2codes = {}
    utt2words = {}
    tot_codeCount = Counter()
    with open(json_fn,"r") as f:
        data_json = json.load(f)
        if "data" in data_json:
            data_json = data_json['data']
    # data_json = data_json[:5000]
    n_utts = min(len(data_json), max_n_utts)
    with open(os.path.join(exp_dir, "data_dict.pkl"), "rb") as f:
        data_dict = pickle.load(f)
    t0 = time.time()
    data_dict_copy = data_dict.copy()
    for utt_index in range(n_utts):
        temp = get_word_ali(data_json, utt_index, level="syllable")
        if temp != None:
            utt2words[utt_index] = temp
            res = get_code_ali_selective(centroid, data_dict, data_json[utt_index]['caption']['wav'] if 'caption' in data_json[utt_index] else data_json[utt_index]['wav'], run_length_encoding, A, b)
            utt2codes[utt_index] = res[0]
            tot_codeCount.update(res[1])
            if 'caption' in data_json[utt_index]:
                if data_json[utt_index]['caption']['wav'] in data_dict_copy:
                    del data_dict_copy[data_json[utt_index]['caption']['wav']]
            else:
                if data_json[utt_index]['wav'] in data_dict_copy:
                    del data_dict_copy[data_json[utt_index]['wav']]
    print("any instance in data_dict that is not in data_json:", data_dict_copy)
    t1 = time.time()
    
    print('Took %.fs to dump %d codes for %d utts'
          % (t1 - t0, sum(list(tot_codeCount.values())), n_utts))
    print(f'There are {len(tot_codeCount)} code clusters')
    return utt2codes, utt2words

def get_word_ali(json_data, index, level="syllable"):
    """
    raw_ali is a string like 'start1__word1__end1 start2__word2__end2 ...'
    """
    raw_ali = json_data[index].get(f'{level}_alignment', None)
    if raw_ali is None:
        return None
    
    data = []
    meta_toks = raw_ali.split()
    for meta_tok in meta_toks:
        toks = meta_tok.split('__')
        if len(toks) == 3:
            data.append((float(toks[0]), float(toks[2]), toks[1]))
    
    if len(data) == 0:
        return None
    else:
        return SparseAlignment(data)


def get_code_ali_selective(centroid, data_dict, wav_id, run_length_encoding=False, A=None, b=None):
    if wav_id not in data_dict:
        return SparseAlignment_code([(0., 0., 0., 0., 0., [-1], torch.randn(768))]), [-1]
    item = data_dict[wav_id]
    seg_center_in_sec = item["locations"]
    boundaries = item['boundaries']
    word_boundaries = item['word_boundaries']
    spf = item['spf']
    if 'seg_feats' in item:
        feats= item["seg_feats"].float()
    else:
        feats = torch.randn(768)
    if "code" in item:
        codes = item['code']
    else:
        assert 'seg_feats' in item, "if code is not in the feat_dict, seg_feats should be in it so we can assign a kmean cluster to each segments"
        if A != None and b != None:
            feats = feats @ A + b
        if centroid.shape[0] == 3: # this is for A40, as we didn't run FAISS on A40
            codes = np.random.randint(0,100,feats.shape[0]).tolist()
        else:
            distances = (torch.sum(feats**2, dim=1, keepdim=True) 
                            + torch.sum(centroid**2, dim=1).unsqueeze(0)
                            - 2 * torch.matmul(feats, centroid.t()))
            codes = torch.min(distances, dim=1)[1].tolist()

        # _, codes = centroid.search(feats.numpy(), 1)
    data = []
    for i, (code, center_in_sec, boundary, w_boundary) in enumerate(zip(codes, seg_center_in_sec, boundaries, word_boundaries)):
        # data.append((center_in_sec-spf/2., center_in_sec+spf/2., boundary[0].item(), boundary[1].item(), w_boundary, [code], feats[i]))
        data.append((center_in_sec-spf/2.+args.shift, center_in_sec+spf/2.+args.shift, boundary[0].item()+args.shift, boundary[1].item()+args.shift, [item+args.shift for item in w_boundary], [code], feats[i]))
    if run_length_encoding:
        # raise NotImplementedError("no need for run length encoding")
        all_codes = torch.tensor([item[-2][0] for item in data]) # all codes [32,32,33,65,178,178,...]
        unique_codes, counts = torch.unique_consecutive(all_codes, return_counts=True)
        collapsed_data = []
        cur_ind = 0
        for code, count in zip(unique_codes, counts):
            # take the (count-1)//2th (start,end) in this repeatition as the start and end
            # so if [1,3,[234]], [5,6,[234]], [7,8,[234]]
            # then the start,end=5,6
            # if [1,3,[234]], [5,6,[234]], [7,8,[234]], [10,11,[234]]
            # then start,end=5,6 also
            ind_use = cur_ind + (count.item()-1)//2
            collapsed_data.append(data[ind_use])
            cur_ind += count
        data = collapsed_data

    return SparseAlignment_code(data), codes


def comp_code_to_wordprec(utt2codes, utt2words, stop_words):
    """
    Return
        code_to_wordprec (dict) : code_to_wordprec[code] is a list of (word,
            precision, num_occ) sorted by precision 
    """
    ts = time.time()
    code_to_nsegs = defaultdict(int)
    code_to_feats = defaultdict(list)
    code_to_wordcounts = defaultdict(Counter)
    for utt_index in utt2codes:
        code_ali = utt2codes[utt_index]    
        word_ali = utt2words[utt_index]
        if word_ali is None:
            continue
        # for code, seg_wordset in align_word_and_code(word_ali, code_ali):
        #     code_to_nsegs[code] += 1
        #     code_to_wordcounts[code].update(seg_wordset)
        # for codes, seg_wordset in align_word_and_code(word_ali, code_ali):
        for codes, seg_wordset, feat in align_word_and_code(word_ali, code_ali):
            for code in codes: #### however, code can repeat!!!
                code_to_nsegs[code] += 1 # denominator is code
                code_to_wordcounts[code].update(seg_wordset)
                code_to_feats[code].append(feat)
    # print('get code_to_wordcounts takes %.fs' % (time.time()-ts))

    code_to_wordprec = dict()     
    n_codes_with_no_words = 0
    for code, nsegs in sorted(code_to_nsegs.items()):
        word_counts = code_to_wordcounts[code]
        word_prec_list = [(word, float(occ)/nsegs, occ) for word, occ \
                          in word_counts.most_common() if word not in stop_words] # co-occur / code occur, this has the problem that if code appear a lot of times within the interval of a word, the resulting precision can be high, run length encoding cannot completely solve this problem as it only tackle adjacent same words, what if within a word, the codes is [12,23,12,44,12,102], in this case 12 con't be collapsed
        n_codes_with_no_words += int(not bool(word_prec_list))
        code_to_wordprec[code] = word_prec_list
    
    print('%d / %d codes mapped to only utterances with empty transcripts' % (
          n_codes_with_no_words, len(code_to_nsegs)))

    # calculate variance of each code cluster
    code_to_variance = dict()
    for code in code_to_feats:
        code_to_variance[code] = torch.stack(code_to_feats[code],dim=0).var(dim=0).mean()
    return code_to_wordprec, code_to_variance
    # return code_to_wordprec

def boundary_mwm(gt, pred):
    n = len(gt) # num of words
    m = len(pred) # num of codes
    sim_mat = np.ones((n,m))
    for i in range(n):
        for j in range(m):
            sim_mat[i,j] = -np.abs(gt[i] - pred[j])
    G = from_biadjacency_matrix(csr_matrix(sim_mat))
    matching = nx.max_weight_matching(G, maxcardinality=True)
    alignment = [_permute(x, sim_mat) for x in matching]
    return [(gt[i], pred[j]) for i, j in alignment]

# def find_boundary_matches_mwm(gt, pred, tolerance, multi_gt_b):
def find_boundary_matches_mwm(gt, pred, tolerance):
    """
    first to max-weight matching to pair gt and pred
    then calculate hit
    """
    gt_len = len(gt)
    pred_len = len(pred)
    match_pred = 0
    match_gt = 0
    matched_pairs = boundary_mwm(gt, pred)
    for gb, pb in matched_pairs:
        if np.abs(gb - pb) <= tolerance:
        # if np.abs(gb - pb) <= tolerance and gb in multi_gt_b:
            match_gt += 1
            match_pred += 1
    return match_gt, match_pred, gt_len, pred_len


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
    # for pred_i in pred:
    #     min_dist = np.abs(gt - pred_i).min()
    #     match_pred += (min_dist <= tolerance)
    # for y_i in gt:
    #     min_dist = np.abs(pred - y_i).min()
    #     match_gt += (min_dist <= tolerance)
    return match_gt, match_pred, gt_len, pred_len


def _permute(edge, sim_mat):
    # Edge not in l,r order. Fix it
    if edge[0] < sim_mat.shape[0]:
        return edge[0], edge[1] - sim_mat.shape[0]
    else:
        return edge[1], edge[0] - sim_mat.shape[0]

def my_max_weight_matching(word_in, code_in, non_word = "non-word", non_code = -1):
    """
    word_in: [(ws, we, w), (ws, we, w), (ws, we, w), ...]
    code_in: [(cs, ce, c), (cs, ce, c), (cs, ce, c), ...]
    they don't need to have the same length

    the output will be max-weight matching aligned word_out, code_out have the same format as inputs, 
    but word_out and word_in will be of the same length if word_in is shorter original, we will fill it with (0,0,'non-word'). if code_in is shorter, we will fill it with (0,0,-1)
 
    """
    n = len(word_in) # num of words
    m = len(code_in) # num of codes

    sim_mat = np.ones((n,m))
    for i in range(n):
        for j in range(m):
            intersection = max(0, min(word_in[i][1], code_in[j][1]) - max(word_in[i][0], code_in[j][0]))
            union = 1. if intersection == 0. else max(word_in[i][1], code_in[j][1]) - min(word_in[i][0], code_in[j][0])
            sim_mat[i,j] = intersection/union
    G = from_biadjacency_matrix(csr_matrix(sim_mat))
    matching = nx.max_weight_matching(G, maxcardinality=True)
    alignment = [_permute(x, sim_mat) for x in matching]
    all_matched_words_ind = [i for i, j in alignment]
    all_matched_codes_ind = [j for i, j in alignment]
    word_out = [word_in[i] for i in all_matched_words_ind]
    code_out = [code_in[j] for j in all_matched_codes_ind]

    code_out += [code_in[j] for j in range(m) if j not in all_matched_codes_ind]
    word_out += [(0,0,non_word) for j in range(m) if j not in all_matched_codes_ind]

    word_out += [word_in[i] for i in range(n) if i not in all_matched_words_ind]
    code_out += [(0,0,non_code) for i in range(n) if i not in all_matched_words_ind]

    assert len(word_out) == len(code_out)
    return word_out, code_out
        




def comp_word_to_coderecall(utt2codes, utt2words, target_words, tolerance, stop_words=STOP_WORDS):
    """
    Compute recall of given words. If `target_words == []`, compute all words.
    
    Return
        word_to_coderecall (dict) : word_to_coderecall[word] is a list of
            (code, recall, num_occ) sorted by recall
    """
    ts = time.time()
    word_to_nsegs = defaultdict(int)
    code_to_nsegs = Counter()
    code_to_feats = defaultdict(list)
    word_to_codecounts = defaultdict(Counter)
    missing_words = Counter()
    IoU = []
    IoT = []
    seg_in_word = []
    CenterDist = []
    CenterIn = []
    coverage_per_sent = []
    left_dist = []
    right_dist = []
    match_gt_count = 0
    match_pred_count = 0
    gt_b_len = 0
    pred_b_len = 0
    for utt_index in utt2codes:
        cur_missing = 0
        cur_words = 0
        code_ali = utt2codes[utt_index]
        word_ali = utt2words[utt_index]
        for _, _, _, _, _, code, feat in code_ali._data:
            code_to_feats[code[0]].append(feat) # code is stored as [code]
        if word_ali is None:
            continue
        gt_boundaries = np.unique([[item[0], item[1]] for item in word_ali.data])
        pred_boundaries = np.unique([item[4] for item in code_ali.data])

        a, b, c, d = find_boundary_matches(gt_boundaries, pred_boundaries, tolerance) 
        match_gt_count += a
        match_pred_count += b
        gt_b_len += c
        pred_b_len += d

        # perform code_ali, word_ali max-weight matching
        # if there are words that doesn't get mapped to any code, map it to code -1, if code doesn't get mapped to any word, map it to 'non-word'
        # the output will be 1-to-1 
        # word_out = [(w_s, w_e, w), (w_s, w_e, w), ...]
        # code_out = [(c_s, c_e, c), (c_s, c_e, c), ...]
        word_in = word_ali._data
        code_in = [(item[2], item[3], item[5][0]) for item in code_ali._data] # TODO make sure code only contains one element, this is a legacy issue
        word_out, code_out = my_max_weight_matching(word_in, code_in)

        for (word_s, word_e, word), (s, e, code) in zip(word_out, code_out):
            if target_words and (word not in target_words):
                continue
            if word in stop_words:
                continue
            if word != "non-word" and code != -1:
                word_to_nsegs[word] += 1
                code_to_nsegs[code] += 1
                word_to_codecounts[word][code] += 1
                cur_words += 1
            else:
                if word != "non-word" and code == -1:
                    word_to_nsegs[word] += 1
                    missing_words[word] += 1
                    cur_missing += 1
                    cur_words += 1
                    continue
                elif code != -1 and word == "non-word":
                    code_to_nsegs[code] += 1
                    continue
                else: # both code == -1 and word == "non-word", this shouldn't happen
                    print("we shouldn't have code == -1 and word =='non-word' aligned to each other!")

            left_dist.append(s - word_s)
            right_dist.append(word_e - s)
            if s >= word_s and e <= word_e:
                cur_iou = (e - s) / (word_e - word_s)
                cur_iot = cur_iou
                seg_in_word.append(1.)
            elif s < word_s and e <= word_e:
                cur_iou = (e - word_s) / (word_e - s)
                cur_iot = (e - word_s) / (word_e - word_s)
                seg_in_word.append(0.)
            elif s >= word_s and e > word_e:
                cur_iou = (word_e - s) / (e - word_s)
                cur_iot = (word_e - s) / (word_e - word_s)
                seg_in_word.append(0.)
            elif s < word_s and e > word_e:
                cur_iou = (word_e - word_s) / (e - s)
                cur_iot = 1.
                seg_in_word.append(0.)
            IoT.append(cur_iot)
            IoU.append(cur_iou)
            word_center = (word_e + word_s)/2.
            seg_center = (e+s)/2.
            CenterDist.append(np.abs(seg_center - word_center))
            CenterIn.append(1. if word_center >= s and word_center <= e else 0.)
        if cur_words > 0:
            coverage_per_sent.append((cur_words - cur_missing)/cur_words)
        else:
            coverage_per_sent.append(0)

    IoU = np.mean(IoU)
    IoT = np.mean(IoT)
    SegInWord = np.mean(seg_in_word)
    CenterDist = np.mean(CenterDist)
    CenterIn = np.mean(CenterIn)
    word_missing_percentage = {}
    total_words = 0
    total_uncovered_words = 0

    for word in word_to_nsegs:
        total_words += word_to_nsegs[word]
        if word in missing_words:
            total_uncovered_words += missing_words[word]
            word_missing_percentage[word] = missing_words[word] / word_to_nsegs[word]
        else:
            word_missing_percentage[word] = 0.
    code_coverage = 1 - total_uncovered_words / total_words

    coverage_per_sent = np.mean(coverage_per_sent)

    word_to_coderecall = dict()
    for word, nsegs in word_to_nsegs.items():
        code_counts = word_to_codecounts[word]
        code_recall_list = [(code, float(occ)/nsegs, occ) for code, occ \
                            in code_counts.most_common() if code != -1] # (word,code) cooccur / word occ, make sure nominator is a fraction of word occ, within the interval of a word, only count the same code once
        word_to_coderecall[word] = code_recall_list

    code_to_wordprec = dict()
    for code, nsegs in code_to_nsegs.items():
        if code == -1:
            continue
        word_precision_list = [(word, float(word_to_codecounts[word][code])/nsegs, word_to_codecounts[word][code]) for word in word_to_codecounts if code in word_to_codecounts[word]]
        word_precision_list.sort(key=lambda x: -x[1]) # rank by co-occurance, but later the pairs are ranked by F1, so this doesn't really matter
        code_to_wordprec[code] = word_precision_list

    # calculate variance of each code cluster
    code_to_variance = dict()
    for code in code_to_feats:
        code_to_variance[code] = torch.stack(code_to_feats[code],dim=0).var(dim=0).mean()
    
    b_prec = match_pred_count / pred_b_len
    b_recall = match_gt_count / gt_b_len
    b_f1 = compute_f1(b_prec, b_recall)
    b_os = b_recall / b_prec - 1.
    b_r1 = np.sqrt((1-b_recall)**2 + b_os**2)
    b_r2 = (-b_os + b_recall - 1) / np.sqrt(2)
    b_r_val = 1. - (np.abs(b_r1) + np.abs(b_r2))/2.
    return word_to_coderecall, code_to_wordprec, code_to_variance, word_to_codecounts, code_to_nsegs, word_to_nsegs, missing_words, word_missing_percentage, code_coverage, coverage_per_sent, IoU, IoT, SegInWord, CenterDist, CenterIn, b_prec, b_recall, b_f1, b_os, b_r_val

def compute_f1(prec, recall):
    return 2*prec*recall / (prec+recall)


########################################################################################
def compute_f1(prec, recall):
    return 2*prec*recall / (prec+recall)

def comp_code_word_f1(code_to_wordprec, word_to_coderecall, min_occ):
    """
    Returns:
        code_to_wordf1 (dict) : code maps to a list of (word, f1, prec, recall, occ)
        word_to_codef1 (dict) : word maps to a list of (code, f1, prec, recall, occ)
    """
    code_to_word2prec = {}
    for code in code_to_wordprec:
        wordprec = code_to_wordprec[code]
        code_to_word2prec[code] = {word : prec for word, prec, _ in wordprec}

    word_to_code2recall = {}
    for word in word_to_coderecall:
        coderecall = word_to_coderecall[word]
        word_to_code2recall[word] = {code : (recall, occ) \
                                     for code, recall, occ in coderecall}

    code_to_wordf1 = defaultdict(list)
    for code in code_to_word2prec:
        for word, prec in code_to_word2prec[code].items():
            recall, occ = word_to_code2recall.get(word, {}).get(code, (0, 0))
            if occ >= min_occ:
                f1 = compute_f1(prec, recall)
                code_to_wordf1[code].append((word, f1, prec, recall, occ))
        code_to_wordf1[code] = sorted(code_to_wordf1[code], key=lambda x: -x[1]) # rank by F1

    word_to_codef1 = defaultdict(list)
    for word in word_to_code2recall:
        for code, (recall, occ) in word_to_code2recall[word].items():
            if occ >= min_occ:
                prec = code_to_word2prec.get(code, {}).get(word, 0)
                f1 = compute_f1(prec, recall)
                word_to_codef1[word].append((code, f1, prec, recall, occ))
        word_to_codef1[word] = sorted(word_to_codef1[word], key=lambda x: -x[1]) # rank by F1

    return code_to_wordf1, word_to_codef1


########################################################################################
class Alignment(object):
    def __init__(self):
        raise

    def __len__(self):
        return len(self.data)

    @property
    def data(self):
        return self._data


class SparseAlignment(Alignment):
    """
    alignment is a list of (start_time, end_time, value) tuples.
    """
    def __init__(self, data, unit=1.):
        self._data = [(s*unit, e*unit, v) for s, e, v in data]

    def __repr__(self):
        return str(self._data)

    def get_segment(self, seg_s, seg_e, empty_word='<SIL>'):
        """
        return words in the given segment.
        """
        seg_ali = self.get_segment_ali(seg_s, seg_e, empty_word)
        return [word for _, _, word in seg_ali.data]

    def get_segment_ali(self, seg_s, seg_e, empty_word=None, contained=False):
        seg_data = []
        if contained:
            is_valid = lambda s, e: (s >= seg_s and e <= seg_e)
        else:
            is_valid = lambda s, e: (max(s, seg_s) < min(e, seg_e))

        for (word_s, word_e, word) in self.data:
            if is_valid(word_s, word_e):
                seg_data.append((word_s, word_e, word))
        if not seg_data and empty_word is not None:
            seg_data = [(seg_s, seg_e, empty_word)]
        return SparseAlignment(seg_data)

    def get_words(self):
        return {word for _, _, word in self.data}

    def has_words(self, check_words):
        """check_words is assumed to be a set"""
        assert(isinstance(check_words, set))
        return bool(self.get_words().intersection(check_words))


class SparseAlignment_code(Alignment):
    """
    alignment is a list of (start_time, end_time, value) tuples.
    """
    def __init__(self, data):
        # v is a list of codes
        # self._data = [(s*unit, e*unit, v) for s, e, v in data]
        self._data = [(s, e, b_s, b_e, w_b, v, feats) for s, e, b_s, b_e, w_b, v, feats in data] # s,e are start end for center frame, b_s and b_e are start end for the segment (all in seconds)

    def __repr__(self):
        return str(self._data)

    def get_segment(self, seg_s, seg_e, empty_code=[-1]):
        """
        return codes in the given segment.
        """
        seg_ali = self.get_segment_ali(seg_s, seg_e, empty_code)
        return [codes[0] for _, _, _, _, _, codes, _ in seg_ali.data], [(codeseg_s, codeseg_e) for _, _, codeseg_s, codeseg_e, _, _, _ in seg_ali.data]

    def get_segment_ali(self, seg_s, seg_e, empty_code=[-1], contained=False):
        seg_data = []
        if contained:
            is_valid = lambda s, e: (s >= seg_s and e <= seg_e)
        else:
            is_valid = lambda s, e: (max(s, seg_s) < min(e, seg_e))
        
        for (code_s, code_e, codeseg_s, codeseg_e, w_b, code, feats) in self.data:
            if is_valid(code_s, code_e):
                seg_data.append((code_s, code_e, codeseg_s, codeseg_e, w_b, code, feats))
        if not seg_data and empty_code is not None:
            seg_data = [(seg_s, seg_e, seg_s, seg_e, w_b, empty_code, feats)]
        return SparseAlignment_code(seg_data)

    def get_codes(self):
        raise NotImplementedError
        return {code for _, _, code in self.data}

    def has_codes(self, check_codes):
        """check_codes is assumed to be a set"""
        raise NotImplementedError
        assert(isinstance(check_codes, set))
        return bool(self.get_codes().intersection(check_codes))


class DenseAlignment(Alignment):
    """
    alignment is a list of values that is assumed to have equal duration.
    """
    def __init__(self, data, spf, offset=0):
        assert(offset >= 0)
        self._data = data
        self._spf = spf
        self._offset = offset
    
    @property
    def spf(self):
        return self._spf

    @property
    def offset(self):
        return self._offset

    def __repr__(self):
        return 'offset=%s, second-per-frame=%s, data=%s' % (self._offset, self._spf, self._data)

    def get_center(self, frm_index):
        return (frm_index + 0.5) * self.spf + self.offset

    def get_segment(self, seg_s, seg_e):
        """
        return words in the given segment
        """
        seg_frm_s = (seg_s - self.offset) / self.spf
        seg_frm_s = int(max(np.floor(seg_frm_s), 0))

        seg_frm_e = (seg_e - self.offset) / self.spf
        seg_frm_e = int(min(np.ceil(seg_frm_e), len(self.data)))
        
        seg_words = self.data[seg_frm_s:seg_frm_e]
        return seg_words

    def get_ali_and_center(self):
        """return a list of (code, center_time_sec)"""
        return [(v, self.get_center(f)) \
                for f, v in enumerate(self.data)]

    def get_sparse_ali(self):
        new_data = list(self.data) + [-1]
        changepoints = [j for j in range(1, len(new_data)) \
                        if new_data[j] != new_data[j-1]]
    
        prev_cp = 0
        sparse_data = []
        for cp in changepoints:
            t_s = prev_cp * self._spf + self._offset
            t_e = cp * self._spf + self._offset
            sparse_data.append((t_s, t_e, new_data[prev_cp]))
            prev_cp = cp
        return SparseAlignment(sparse_data)


##############################
# Transcript Post-Processing #
##############################

def align_sparse_to_dense(sp_ali, dn_ali, center_to_range):
    """
    ARGS:
        sp_ali (SparseAlignment):
        dn_ali (DenseAlignment):
    """
    ret = []
    w_s_list, w_e_list, w_list = zip(*sp_ali.data)
    w_sidx = 0  # first word that the current segment's start is before a word's end
    w_eidx = 0  # first word that the current segment's end is before a word's start
    for code, cs in dn_ali.get_ali_and_center():
        ss, es = center_to_range(cs)
        while w_sidx < len(w_list) and ss > w_e_list[w_sidx]:
            w_sidx += 1
        while w_eidx < len(w_list) and es > w_s_list[w_eidx]:
            w_eidx += 1
        seg_wordset = set(w_list[w_sidx:w_eidx]) if w_eidx > w_sidx else {'<SIL>'}
        ret.append((code, seg_wordset))
    return ret

def align_word_and_code(word_ali, code_ali):
    """
    ARGS:
        word_ali (SparseAlignment):
        code_ali (SparseAlignment):
    """
    ret = []
    w_s_list, w_e_list, w_list = zip(*word_ali.data)
    # c_s_list, c_e_list, c_list = zip(*code_ali.data)
    c_s_list, c_e_list, c_list, f_list = zip(*code_ali.data)
    w_sidx = 0  # first word that the current segment's start is before a word's end
    w_eidx = 0  # first word that the current segment's end is before a word's start
    # for ss, es, code in code_ali:
    # for ss, es, code in zip(c_s_list, c_e_list, c_list):
    for ss, es, code, feat in zip(c_s_list, c_e_list, c_list, f_list):
        while w_sidx < len(w_list) and ss > w_e_list[w_sidx]:
            w_sidx += 1
        while w_eidx < len(w_list) and es > w_s_list[w_eidx]:
            w_eidx += 1
        seg_wordset = set(w_list[w_sidx:w_eidx]) if w_eidx > w_sidx else {'<SIL>'}
        # ret.append((code, seg_wordset))
        ret.append((code, seg_wordset, feat))
    return ret

def print_code_to_word_prec(code_to_wordprec, prec_threshold=0.35,
                            num_show=3, show_all=True):
    n_codes = len(code_to_wordprec.keys())
    n_codes_above_prec_threshold = 0
            
    for code in sorted(code_to_wordprec.keys()):
        wordprec = code_to_wordprec[code]
        if not len(wordprec):
            continue
        (top_word, top_prec, _) = wordprec[0]
        above_prec_threshold = (top_prec >= prec_threshold)
        if above_prec_threshold:
            n_codes_above_prec_threshold += 1
            
        if show_all or above_prec_threshold:
            tot_occ = sum([occ for _, _, occ in wordprec])
            # show top-k
            msg = "%s %4d (#words=%5d, occ=%5d): " % (
                "*" if above_prec_threshold else " ", 
                code, len(wordprec), tot_occ)
            for word, prec, _ in wordprec[:num_show]:
                res = "%s (%5.2f)" % (word, prec)
                msg += " %-25s|" % res
            print(msg)
    
    print(('Found %d / %d (%.2f%%) codes with a word detector with' 
           'prec greater than %f.') % (
            n_codes_above_prec_threshold, n_codes, 
            n_codes_above_prec_threshold / n_codes * 100, prec_threshold)) 

def print_word_by_code_recall(word_to_coderecall, num_show=3):
    for word in sorted(word_to_coderecall):
        tot_occ = sum([o for _, _, o in word_to_coderecall[word]])
        print("%-15s (#occ = %4d)" % (word, tot_occ),
              [('%4d' % c, '%.2f' % r) for c, r, _ in word_to_coderecall[word][:num_show]])


def print_code_stats_by_f1(code_to_wordf1, code_ranks_show=range(10),
                           num_word_show=2):
    print("##### Showing ranks %s" % str(code_ranks_show))
    codes = sorted(code_to_wordf1.keys(), 
                   key=lambda x: (-code_to_wordf1[x][0][1] if len(code_to_wordf1[x]) else 0))
    print('%3s & %4s & %10s & %6s & %6s & %6s & %5s'
          % ('rk', 'code', 'word', 'F1', 'Prec', 'Recall', 'Occ'))
    for rank in code_ranks_show:
        if rank >= len(codes):
            continue
        code = codes[rank]
        msg = '%3d & %4d' % (rank+1, code)
        for word, f1, prec, recall, occ in code_to_wordf1[code][:num_word_show]:
            msg += ' & %10s & %6.2f & %6.2f & %6.2f & %5d' % (
                    word.lower(), f1*100, prec*100, recall*100, occ)
        msg += ' \\\\'
        print(msg)

def print_clustering_purity_nmi(code_to_nsegs, word_to_nsegs, code_to_wordprec, empty_code=-1):
    '''
    code_to_nsegs {code: count_of_this_code}
    word_to_nsegs: {word: count_of_this_word}
    code_to_wordprec: {code: [(word, cond_prob (word given code), co-occ), ...]}, rank by co-occ, doesn't contain -1
    '''
    from scipy import stats

    if -1 in code_to_nsegs.copy():
        del code_to_nsegs[-1]

    codes = [code for code in code_to_nsegs]
    total_n_codes = sum(code_to_nsegs.values())
    purity = sum(max(item[-1] for item in code_to_wordprec[code]) for code in codes if code != -1 and len([item[-1] for item in code_to_wordprec[code]])!=0)/total_n_codes # in the word discovery IS2022 paper, we get rid of -1 cluster when calculating purity, in this case, we do not
    code_purity = {code: max(item[-1] for item in code_to_wordprec[code])/code_to_nsegs[code] for code in codes if len([item[-1] for item in code_to_wordprec[code]])!=0}

    print(f"purity: {purity:.4f}")
    return code_purity


def plot_and_save(c2wf, c2var, basename):
    """
    c2wf, or code to word precision 
     is a dict, key is code, value is a list of (word, f1, prec, recall, occ), sorted by f1
    
    c2var, or code to cluster variance
     is a dict, key is code, value is mean variance of this cluster
    """
    from matplotlib import pyplot as plt
    import seaborn as sns
    import matplotlib
    matplotlib.style.use("ggplot")
    os.makedirs(f"../pics/{basename}", exist_ok=True)
    var_f1 = np.array([[c2var[code].item(), c2wf[code][0][1]] for code in c2wf if len(c2wf[code]) > 0])
    plt.figure()
    plt.xlabel("var")
    plt.ylabel("f1")
    plot = sns.scatterplot(x = var_f1[:,0], gt = var_f1[:,1])
    plot.figure.savefig(f"../pics/{basename}/var_f1.png")
    

    plt.figure()
    var_occ = np.array([[c2var[code].item(), c2wf[code][0][-1]] for code in c2wf if len(c2wf[code]) > 0])
    plt.xlabel("var")
    plt.ylabel("occ")
    plot = sns.scatterplot(x = var_occ[:,0], gt = var_occ[:,1])
    plot.figure.savefig(f"../pics/{basename}/var_occ.png")


    plt.figure()
    var_prec = np.array([[c2var[code].item(), c2wf[code][0][2]] for code in c2wf if len(c2wf[code]) > 0])
    plt.xlabel("var")
    plt.ylabel("prec")
    plot = sns.scatterplot(x = var_prec[:,0], gt = var_prec[:,1])
    plot.figure.savefig(f"../pics/{basename}/var_prec.png")

    plt.figure()
    var_recall = np.array([[c2var[code].item(), c2wf[code][0][3]] for code in c2wf if len(c2wf[code]) > 0])
    plt.xlabel("var")
    plt.ylabel("recall")
    plot = sns.scatterplot(x = var_recall[:,0], gt = var_recall[:,1])
    plot.figure.savefig(f"../pics/{basename}/var_recall.png")

def plot_missing_words(word_to_nsegs, missing_words, word_missing_percentage, topk, basename):
    from matplotlib import pyplot as plt
    import seaborn as sns
    import matplotlib
    matplotlib.style.use("ggplot")
    os.makedirs(f"../pics/{basename}", exist_ok=True)
    word_freq = sorted(word_to_nsegs.items(), key=lambda x: x[1], reverse=True)
    missing_freq = sorted(missing_words.items(), key=lambda x: x[1], reverse=True)
    percentage = sorted(word_missing_percentage.items(), key=lambda x: x[1], reverse=True)
    plt.figure(figsize=(24,12))
    plt.title("Word Occurance")
    plt.ylabel("Counts")
    temp = word_freq[:topk]
    plt.xticks(range(len(temp)), [item[0] for item in temp])
    plt.bar(range(len(temp)), [item[1] for item in temp])
    plt.savefig(f"../pics/{basename}/word_occurance.png")
    
    plt.figure(figsize=(24,12))
    plt.title("Word Missing Count")
    plt.ylabel("Counts")
    temp = missing_freq[:topk]
    plt.xticks(range(len(temp)), [item[0] for item in temp])
    plt.bar(range(len(temp)), [item[1] for item in temp])
    plt.savefig(f"../pics/{basename}/word_missing.png")
    
    plt.figure(figsize=(24,12))
    plt.title("Word Missing Percentage")
    plt.ylabel("Percentage %")
    temp = percentage[:topk]
    plt.xticks(range(len(temp)), [item[0] for item in temp])
    plt.bar(range(len(temp)), [item[1] for item in temp])
    plt.savefig(f"../pics/{basename}/word_missing_percentage.png")

def threshold_by_var(c2wf, c2var, threshold):
    c2wf_th = dict()
    for code in c2var:
        if c2var[code] <= threshold:
            c2wf_th[code] = c2wf[code]
    return c2wf_th


def print_word_stats_by_f1(word_to_codef1, word_ranks_show=range(10),
                           num_code_show=3):
    print("##### Showing ranks %s" % str(word_ranks_show))
    words = sorted(word_to_codef1.keys(), 
                   key=lambda x: (-word_to_codef1[x][0][1] if len(word_to_codef1[x]) else 0))
    print('%3s & %15s & %4s & %6s & %6s & %6s & %5s'
          % ('rk', 'word', 'code', 'F1', 'Prec', 'Recall', 'Occ'))
    for rank in word_ranks_show:
        if rank >= len(words):
            continue
        word = words[rank]
        for code, f1, prec, recall, occ in word_to_codef1[word][:num_code_show]:
            if occ < 100:
                continue
            msg = '%3d & %15s' % (rank+1, word.lower())
            msg += ' & %4d & %6.2f & %6.2f & %6.2f & %5d \\\\' % (
                    code, f1*100, prec*100, recall*100, occ)
            print(msg)
    

def count_high_f1_words(word_to_codef1, f1_threshold=0.5, verbose=True):
    count = 0 
    for word in word_to_codef1.keys():
        if len(word_to_codef1[word]) and (word_to_codef1[word][0][1] >= f1_threshold):
            count += 1
    if verbose:
        print('%d / %d words with an F1 score >= %s'
              % (count, len(word_to_codef1), f1_threshold))
    return count
    
def count_high_f1_codes(word_to_codef1, f1_threshold=0.5, verbose=True):
    count = 0 
    for word in word_to_codef1.keys():
        if len(word_to_codef1[word]) and (word_to_codef1[word][0][1] >= f1_threshold):
            count += 1
    if verbose:
        print('%d / %d codes with an F1 score >= %s'
              % (count, len(word_to_codef1), f1_threshold))
    return count
def count_high_precision_recall_codes(code_to_wordf1, threshold=0.5, verbose=True):
    prec_count = 0 
    recall_count = 0 
    for code in code_to_wordf1.keys():
        if len(code_to_wordf1[code]) and (code_to_wordf1[code][0][2] >= threshold):
            prec_count += 1
        if len(code_to_wordf1[code]) and (code_to_wordf1[code][0][3] >= threshold):
            recall_count += 1
    if verbose:
        print('%d / %d codes with an precision score >= %s'
              % (prec_count, len(code_to_wordf1), threshold))
        print('%d / %d codes with an recall score >= %s'
              % (recall_count, len(code_to_wordf1), threshold))
    return prec_count, recall_count
def compute_topk_avg_f1(word_to_codef1, k=250, verbose=True):
    f1s = [word_to_codef1[word][0][1] for word in word_to_codef1 \
           if len(word_to_codef1[word])]
    top_f1s = sorted(f1s, reverse=True)[:k]
    
    if verbose:
        print('avg F1 = %.2f%% for top %d words; %.2f%% for all %d words'
              % (100*np.mean(top_f1s), len(top_f1s), 100*np.mean(f1s), len(f1s)))
    return 100*np.mean(top_f1s)


def print_analysis(code_to_wordprec, word_to_coderecall,
                   code_norms, high_prec_words, min_occ, rank_range):
    print_code_to_word_prec(code_to_wordprec, prec_threshold=0.35,
                            num_show=3, show_all=True)
    print_word_by_code_recall(word_to_coderecall, num_show=3)

    code_to_wordf1, word_to_codef1 = comp_code_word_f1(
            code_to_wordprec, word_to_coderecall, min_occ=min_occ)
    print_code_stats_by_f1(code_to_wordf1, rank_range)
    print_word_stats_by_f1(word_to_codef1, rank_range)
    count_high_f1_words(word_to_codef1, f1_threshold=0.5)
    compute_topk_avg_f1(word_to_codef1, k=250)


def cluster_analysis2(w2cf, c2wf, word_to_nsegs, code_to_nsegs, threshold=0.5):
    good_words = {} # word:#occ
    good_codes = {} # code:#occ
    total_words = sum(list(word_to_nsegs.values()))
    total_codes = sum(list(code_to_nsegs.values()))
    for w in w2cf:
        if len(w2cf[w]) >= 1 and w2cf[w][0][1] >= threshold:
            good_words[w] = {"detected": w2cf[w][0][-1], "total":word_to_nsegs[w]}
    for c in c2wf:
        if len(c2wf[c]) >= 1 and c2wf[c][0][1] >= threshold:
            good_codes[c] = {"detected": c2wf[c][0][-1], "total": code_to_nsegs[c]}
    print(f"good words/total words: {sum([item['detected'] for item in good_words.values()])}/{total_words} -> {sum([item['detected'] for item in good_words.values()])/total_words}")
    print(f"good codes/total codes: {sum([item['detected'] for item in good_codes.values()])}/{total_codes} -> {sum([item['detected'] for item in good_codes.values()])/total_codes}")

def get_detection_stats(centroid, max_n_utts, exp_dir, json_fn, tolerance, stop_words=STOP_WORDS, min_occ=1, run_length_encoding=False, A=None, b=None):
    utt2codes, utt2words= prepare_data(centroid, json_fn, max_n_utts, exp_dir, run_length_encoding = run_length_encoding, A=A, b=b)
    w2cr, c2wp, c2var, word_to_codecounts, code_to_nsegs, word_to_nsegs, missing_words, word_missing_percentage, code_coverage, coverage_per_sent, IoU, IoT, SegInWord, CenterDist, CenterIn, b_prec, b_recall, b_f1, b_os, b_r_val = comp_word_to_coderecall(utt2codes, utt2words, [], tolerance, stop_words)
    
    c2wf, w2cf = comp_code_word_f1(c2wp, w2cr, min_occ=min_occ)
    return c2wp, c2var, w2cr, c2wf, w2cf, word_to_codecounts, code_to_nsegs, word_to_nsegs, missing_words, word_missing_percentage, code_coverage, coverage_per_sent, IoU, IoT, SegInWord, CenterDist, CenterIn, b_prec, b_recall, b_f1, b_os, b_r_val
    
print("\nI am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_json", type=str, default="/data1/scratch/coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy_with_alignments.json", help="the fn of force alignment json file")
parser.add_argument("--exp_dir", type=str, default="/scratch/cluster/pyp/exp_pyp/discovery/word_unit_discovery/disc-23/curFeats_mean_0.9_7_forceAlign")
parser.add_argument("--k", type=int, default=4096)
parser.add_argument("--run_length_encoding", action="store_true", default=False, help="if True, collapse all adjacent same code into one code; if False, use the original implementation, which, when calculate word2code_recall, it collapse all same code within the same word into one code. and when calculating code2word_precision, it doesn't do anything, so if a code appears 10 times (within the interval of a word), this are accounted as coappearing 10 times ")
parser.add_argument("--iou", action="store_true", default=False, help="wether or not evaluate the intersection over union, center of mass distance, center of mass being in segment percentage")
parser.add_argument("--max_n_utts", type=int, default=200000, help="total number of utterances to study, there are 25020 for SpokenCOCO, so if the number is bigger than that, means use all utterances")
parser.add_argument("--topk", type=int, default=30, help="show stats of the topk words in hisst plot")
parser.add_argument("--tolerance", type=float, default=0.02, help="tolerance of word boundary match")
parser.add_argument("--shift", type=float, default=0)

args = parser.parse_args()
kmeans_dir = f"{args.exp_dir}/kmeans_models/CLUS{args.k}/centroids.npy"
if not os.path.isfile(kmeans_dir):
    print("kmeans centroid not found")
    centroid = torch.randn((3, 768))
else:
    centroid = torch.from_numpy(np.load(kmeans_dir))
    print("kmeans centroids shape: ", centroid.shape)
A = None
b = None


c2wp, c2var, w2cr, c2wf, w2cf, word_to_codecounts, code_to_nsegs, word_to_nsegs, missing_words, word_missing_percentage, code_coverage, coverage_per_sent, IoU, IoT, SegInWord, CenterDist, CenterIn, b_prec, b_recall, b_f1, b_os, b_r_val = get_detection_stats(centroid, args.max_n_utts, tolerance= args.tolerance, exp_dir=args.exp_dir, json_fn=args.data_json, min_occ=1, run_length_encoding=args.run_length_encoding, A=A, b=b)



print(f"AScore: {2./(1/IoU+1/code_coverage):.4f}")
print(f"IoU: {IoU:.4f}")
print(f"IoT: {IoT:.4f}")
print(f"Percentage that the segment falls within the word interval: {SegInWord*100:.2f}")
print(f"Average distance (in seconds) between segment center and word center: {CenterDist:.4f}")
print(f"Percentage that word centers fall into the code segment: {CenterIn*100:.2f}%")
print(f"code coverage (average over all *words*): {code_coverage:.4f}")
print(f"code coverage (average over all *word types*): {sum([1-item for item in word_missing_percentage.values()])/len(word_missing_percentage):.4f}")
print(f"coverage per sentence: {coverage_per_sent:.4f}")
print(f"boundary precision: {b_prec:.4f}")
print(f"boundary recall: {b_recall:.4f}")
print(f"boundary F1: {b_f1:.4f}")
print(f"boundary over-segmentation: {b_os:.4f}")
print(f"boundary R value: {b_r_val:.4f}")
code_purity = print_clustering_purity_nmi(code_to_nsegs, word_to_nsegs, c2wp)




count_high_f1_words(w2cf, f1_threshold=0.5)
count_high_f1_codes(c2wf, f1_threshold=0.5)
prec_count, recall_count = count_high_precision_recall_codes(c2wf, threshold=0.5)
top_word_avg_f1 = compute_topk_avg_f1(w2cf, k=250)
cluster_analysis2(w2cf, c2wf, word_to_nsegs, code_to_nsegs)
print_word_stats_by_f1(w2cf, range(200), num_code_show=1)

