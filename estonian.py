import argparse
import torch
import os
import pickle
import json
import soundfile as sf
import tqdm
import time
import numpy as np
import torchaudio
import torchaudio.transforms as at
from itertools import groupby
from operator import itemgetter

from mincut import mincut
from models import audio_encoder


def cls_attn_seg_feats(cls_attn_weights, threshold, spf):
    # return a list of features that are segmented by cls attn weights
    cls_attn_weights = torch.from_numpy(cls_attn_weights)
    threshold_value = torch.quantile(cls_attn_weights, threshold, dim=-1, keepdim=True) # [n_h, T]
    important_idx = torch.where((cls_attn_weights >= threshold_value).float().sum(0) > 0)[0].numpy()
    boundaries = []
    boundaries_all = []
    boundaries_ex1 = []
    for k, g in groupby(enumerate(important_idx), lambda ix : ix[0] - ix[1]):
        seg = list(map(itemgetter(1), g))
        t_s, t_e = seg[0], min(seg[-1]+1, cls_attn_weights.shape[-1])
        if len(seg) > 1:
            boundaries_all.append([t_s, t_e])
            boundaries_ex1.append([t_s, t_e])
        else:
            boundaries_all.append([t_s, t_e])
    
    if len(boundaries_ex1) == 0:
        boundaries = boundaries_all
    else:
        boundaries = boundaries_ex1

    boundaries_in_sec = np.unique([[t_s*spf, t_e*spf] for t_s, t_e in boundaries]).tolist()
    
    return boundaries_in_sec

def mincut_parallel(audio_len_sec, ali, feat, spf, wav_fn, segment_method):
    if args.secPerSyllable >= 10000: # probably the best guess of num_sullables, actually n_syl + n_sil blocks
        num_syllable = len(ali)
    else:
        num_syllable = int(np.ceil(audio_len_sec / args.secPerSyllable)) if args.secPerSyllable > 0 else len(ali) - int(args.secPerSyllable) # if negative number, means number of ground truth - that number

    ssm = feat@feat.transpose(1,0)
    ssm = ssm - np.min(ssm) + 1e-7
    seg_boundary_frame = mincut.min_cut(ssm, num_syllable+1) # +1 for the algo
    seg_boundary_frame_pairs_orig = [[l,r] for l, r in zip(seg_boundary_frame[:-1], seg_boundary_frame[1:])] # 
    seg_boundary_frame_pairs = [item for item in seg_boundary_frame_pairs_orig if item[1]-item[0] > 2]
    if len(seg_boundary_frame_pairs)==0: # this shouldn't happen though
        seg_boundary_frame_pairs = seg_boundary_frame_pairs_orig

    if "merge" in segment_method.lower() and len(seg_boundary_frame_pairs) >= 3:
        seg_boundary_frame_pairs = seg_boundary_frame_pairs_orig
        merge_thres = float(segment_method.split("-")[-1])
        all_feat = [feat[round(l):round(r)].mean(0) for l,r in seg_boundary_frame_pairs]
        all_sim = [np.dot(l,r)/(np.linalg.norm(l)*np.linalg.norm(r)) for l,r in zip(all_feat[:-1], all_feat[1:])]
        min_id = np.argmax(all_sim)
        while all_sim[min_id] >= merge_thres and len(seg_boundary_frame_pairs) >= 3:
            l_merge, r_merge = seg_boundary_frame_pairs[min_id], seg_boundary_frame_pairs[min_id+1]
            seg_boundary_frame_pairs = [pair for i, pair in enumerate(seg_boundary_frame_pairs) if i != min_id and i != min_id+1]
            seg_boundary_frame_pairs.insert(min_id, [l_merge[0], r_merge[1]])
            all_feat = [feat[round(l):round(r)].mean(0) for l,r in seg_boundary_frame_pairs]
            all_sim = [np.dot(l,r)/(np.linalg.norm(l)*np.linalg.norm(r)) for l,r in zip(all_feat[:-1], all_feat[1:])]
            min_id = np.argmax(all_sim)

    word_boundaries = [[l*spf,r*spf] for l, r in seg_boundary_frame_pairs]
    word_boundaries = np.unique(word_boundaries).tolist()
    word_boundaries = word_boundaries[1:-1]

    return word_boundaries, ali, wav_fn

def compute_f1(prec, recall):
    return 2*prec*recall / (prec+recall)

def find_boundary_matches(gt, pred, syl_list, tolerance):
    """
    gt: list of ground truth boundaries
    pred: list of predicted boundaries
    all in seconds
    """
    gt_len = len(gt)
    pred_len = len(pred)
    gt_pointer = 0
    pred_pointer = 0
    match_pred = 0
    match_gt = 0
    ## below method will avoid counting same gt or pred boundaries more than once
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

print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_dir", type=str)
parser.add_argument("--dataset", type=str, default='spokencoco')
parser.add_argument("--seg_fn", type=str, default='no', help="path to the segmentation (i.e. boundaries) file, if not provided, use do segmentation and feature extraction on the fly")
parser.add_argument("--snapshot", type=str, default='best', help='which model snapshot to use, best means best_boundle.pth, can also pass number x, say 24, then will take snapshot_24.pth')
parser.add_argument("--valid_pkl_fn", type=str, default="/home/pyp/More-Discovery/valid_estonian.pkl")
parser.add_argument("--valid_wav_path", type=str, default="/data3/scratch/pyp/estonian/SKK0_WAV")
parser.add_argument("--test_root", type=str, default="/data3/scratch/pyp/estonian/okko")
parser.add_argument("--max_iter", type=int, default=200)
parser.add_argument("--save_root", type=str, default="/data2/scratch/pyp/discovery/word_unit_discovery/")
parser.add_argument("--reduce_method", type=str, default="mean", choices=['mean', 'max', 'median', 'weightedmean'])
parser.add_argument("--layer", type=int, default=7, help="where attn weights are coming from, as for features, if feats_type==preFeats, and feature comes from previous layer of tgt_layer_for_attn, otherwise, feature comes from the same layer")
parser.add_argument("--level2", action="store_true", default=False, help="if True, use feats and atten weights from level2 (not avaliable for models that only has one level of w2v2)")
parser.add_argument("--segment_method", type=str, default=None, help="example choices=['clsAttn-0.7', 'minCut', 'forceAlign', 'uniform', 'minCutMerge-0.5']")
parser.add_argument("--spf", type=float, default=0.02)
parser.add_argument("--secPerSyllable", type=float, help="second per segment, given audio_len_sec, we get K in the min cut algorithm by audio_len_sec/secPerSyllable, 0 means use the ground truth")
parser.add_argument("--n_proc", type=int, default=32, help="number of parallel process for min cut segmentation")
parser.add_argument("--tolerance", type=float, default=0.05)
args = parser.parse_args()

feats_type = args.dataset + "_" + args.segment_method + "_" + "secPerSyllable" + str(args.secPerSyllable if args.secPerSyllable > 0 else int(args.secPerSyllable)) + "_" + str(args.layer) + "_" + args.reduce_method + "_" + "snapshot" + args.snapshot

save_root = os.path.join(args.save_root, feats_type)
print("data save at: ", save_root)
os.makedirs(save_root, exist_ok=True)
print(args)
if not os.path.isdir(args.exp_dir) and "hubert" not in args.exp_dir.lower():
    raise RuntimeError(f"{args.exp_dir} does not exist!!")

data_start_time = time.time()

############ hard coded a bunch of things ############

if args.dataset == "estonianval":
    with open(args.valid_pkl_fn, "rb") as f:
        all_data = pickle.load(f)
    wav_root = args.valid_wav_path
    start_list = []
    end_list = []
    gt_list = []
    syl_list = []
    wav_list = []
    actual_wav_list = []
    for wav_key in all_data:
        actual_wav_list.append(wav_key)
        wav_list.append("_".join(wav_key.split("_")[:-1]) + ".wav")
        start_list.append(all_data[wav_key]["start"][0])
        end_list.append(all_data[wav_key]["end"][-1])
        temp = [item - all_data[wav_key]["start"][0] for item in np.unique(all_data[wav_key]["start"]+all_data[wav_key]["end"]).tolist()]
        gt_list.append(temp)
        syl_list.append(all_data[wav_key]["text"])
        # if len(gt_list) > 99:
        #     break
    print(f"estonian validation have in total {len(wav_list)} examples")

else:
    start_list = []
    end_list = []
    import scipy.io
    mat = scipy.io.loadmat(os.path.join(args.test_root, 'SKK_anno.mat'))
    wav_root = os.path.join(args.test_root, "SKK0_utterances")
    wav_list = [item[0].item().split("/")[-1] for item in mat['anno'][0][0][0]]
    gt_list1 = [item[0].astype(np.float32) for item in mat['anno'][0][0][7]]
    gt_list2 = [item[0].astype(np.float32) for item in mat['anno'][0][0][8]]
    syl_list = [item1[0] for item1 in mat['anno'][0][0][8]]
    final_wav_list = []
    final_syl_list = []
    final_gt_list = []
    for wav, syl, gt1, gt2 in zip(wav_list, syl_list, gt_list1, gt_list2):
        if len(gt1) > 0 and len(gt2)>0:
            assert len(gt1) == len(gt2), f"len(gt1): {len(gt1)}, len(gt1): {len(gt2)}"
            cur_gt_list = [[item1[0], item2[0]] for item1, item2 in zip(gt1, gt2)]
            final_wav_list.append(wav)
            final_syl_list.append([item[0].item() for item in syl])
            final_gt_list.append(np.unique(cur_gt_list))

    gt_list, syl_list, wav_list = final_gt_list, final_syl_list, final_wav_list


for gt, syl, wav in zip(gt_list, syl_list, wav_list):
    assert len(gt) == len(syl) + 1, f"len(gt): {len(gt)}, len(syl): {len(syl)}"


locF_temp = []
j = 0
temp_data_dict = {}
data_dict = {}
missing_ali = 0
temp_fn = os.path.join(args.save_root, f'{args.dataset}_{args.snapshot}_{args.layer}_data_dict.pkl')
if os.path.isfile(temp_fn) and "clsAttn" not in args.segment_method:
    print(temp_fn, "exists, load the existing one" )
    with open(temp_fn, "rb") as f:
        temp_data_dict = pickle.load(f)
    print(f"len of the temp_data_dict: {len(temp_data_dict)}")
else:
    if 'hubert' in args.exp_dir.lower():
        from transformers import Wav2Vec2Processor, HubertModel
        model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        model = model.cuda()
        model = model.eval()
        model_args = None
    else:
        ########################## setup model ##########################
        with open(os.path.join(args.exp_dir, "args.pkl"), "rb") as f:
            model_args = pickle.load(f)
        model = audio_encoder.AudioEncoder(model_args)
        if "best" in args.snapshot:
            bundle = torch.load(os.path.join(args.exp_dir, "best_bundle.pth"))
        else:
            snapshot = int(args.snapshot)
            bundle = torch.load(os.path.join(args.exp_dir, f"snapshot_{snapshot}.pth"))

        if "dual_encoder" in bundle:
            model.carefully_load_state_dict(bundle['dual_encoder'], load_all=True)
        elif "audio_encoder" in bundle:
            model.carefully_load_state_dict(bundle['audio_encoder'], load_all=True)
        else:
            model.carefully_load_state_dict(bundle['model'], load_all=True)

        model.eval()
        model = model.cuda()
        ########################## setup model ##########################
    for j, wav_key in enumerate(tqdm.tqdm(wav_list)):
        assert "estonian" in args.dataset
        if "val" in args.dataset:
            metadata = torchaudio.info(os.path.join(wav_root, wav_key))
            sr = metadata.sample_rate
            start_frame, end_frame = int(round(sr*start_list[j])), int(round(sr*end_list[j]))
            audio, sr = torchaudio.load(filepath=os.path.join(wav_root, wav_key), frame_offset=max(0,start_frame-1), num_frames=end_frame-start_frame, normalize=True)
            audio = audio.squeeze(0).float().numpy()
            wav_key = actual_wav_list[j]
            assert len(audio.shape) == 1, audio.shape
        else:    
            audio, sr = sf.read(os.path.join(wav_root, wav_key), dtype = 'float32')
            # print(sr) # it's always 44100
        if sr != 16000:
            audio = at.Resample(sr, 16000, dtype=torch.float32)(torch.from_numpy(audio).unsqueeze(0)).squeeze(0).numpy()
            sr = 16000
        assert len(audio.shape) == 1, audio.shape
        with torch.no_grad():
            if "hubert" in args.exp_dir.lower():
                out = model(input_values=torch.from_numpy(audio).unsqueeze(0).cuda(), output_hidden_states=True, return_dict=True)
                feat = out['hidden_states'][args.layer][0].cpu().float().numpy()
                attn_weights = np.random.randn(feat.shape[-1])
            else:
                out = model(torch.from_numpy(audio).unsqueeze(0).cuda(), padding_mask=None, mask=False, need_attention_weights=True, tgt_layer=args.layer, pre_feats= False, level2=False)
                if model_args != None and not model_args.use_audio_cls_token: # to handle SC finetuned HuBERT (disc-63 on RTX)
                    feat = out['features'].squeeze(0).cpu().float().numpy()
                    cls_attn_weights = np.random.randn(12, len(feat))
                else:
                    feat = out['features'].squeeze(0)[1:].cpu().float().numpy()
                    cls_attn_weights = out['attn_weights'].squeeze(0)[:, 0, 1:].cpu().float().numpy() # [num_heads, tgt_len, src_len] -> [n_h, T]
        spf = audio.shape[0]/sr/feat.shape[0]
        if "clsAttn" in args.segment_method:
            threshold = float(args.segment_method.split("-")[-1])
            pred = cls_attn_seg_feats(cls_attn_weights, threshold, spf)
            data_dict[wav_key] = {"pred": pred, "gt": gt_list[j]}
        else:
            temp_data_dict[wav_key] = {"feat": feat, "audio_len_sec": audio.shape[0]/sr, "spf": spf, "ali": gt_list[j]}
        # if j >= 100:
        #     break
    if "clsAttn" not in args.segment_method:
        print("save the extracted temp_data_dict at ", temp_fn)
        with open(temp_fn, "wb") as f:
            pickle.dump(temp_data_dict, f)



if "minCut" in args.segment_method and args.n_proc > 1:
    import joblib
    parallizer = joblib.Parallel(n_jobs=args.n_proc, max_nbytes=None, verbose=2)
    res = parallizer(joblib.delayed(mincut_parallel)(temp_data_dict[wav_fn]['audio_len_sec'], temp_data_dict[wav_fn]['ali'], temp_data_dict[wav_fn]['feat'], temp_data_dict[wav_fn]['spf'], wav_fn, args.segment_method) for wav_fn in list(temp_data_dict.keys()))
    for item in res:
        # print(item[0])
        data_dict[item[-1]] = {"pred": item[0], "gt": item[1]} 
else:
    raise NotImplementedError



tolerance=0.05
shifts = np.arange(-0.06, 0.031, 0.005)
shift_res = {}
# save the segmentation result when calculating metrics, just in case we want to reuse
for shift in shifts:
    match_gt_count = 0
    match_pred_count = 0
    gt_b_len = 0
    pred_b_len = 0
    for j, wav_key in enumerate(data_dict):
        pred_b = np.unique(data_dict[wav_key]['pred']).tolist()
        pred_b = [bb + shift for bb in pred_b]
        gt_b = data_dict[wav_key]['gt']
        if syl_list[j][0] == "h#":
            gt_b = gt_b[1:]
        if syl_list[j][-1] == "h#":
            gt_b = gt_b[:-1]
        if not len(gt_b):
            continue
        cur_start, cur_end = gt_b[0], gt_b[-1]
        while len(pred_b) and pred_b[0] <= cur_start:
            pred_b = pred_b[1:]
        while len(pred_b) and pred_b[-1] >= cur_end:
            pred_b = pred_b[:-1]
        pred_b = [cur_start] + pred_b + [cur_end]
        pred_b = np.unique(pred_b).tolist()
        a, b, c, d = find_boundary_matches(gt_b, pred_b, syl_list[j], tolerance)
        
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

all_f1 = [val[3] for val in shift_res.values()]
# all_f1 = [val[4] for val in shift_res.values()]
ind = np.argmax(all_f1)
best_shift = list(shift_res.keys())[ind]

print("shift:", best_shift)
print("prec:", shift_res[best_shift][0])
print("rec: ", shift_res[best_shift][1])
print("os:", shift_res[best_shift][2])
print("f1: ", shift_res[best_shift][3])
print("R val: ", shift_res[best_shift][4])
