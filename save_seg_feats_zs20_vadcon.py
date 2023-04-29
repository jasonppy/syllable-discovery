import argparse
import torch
import os
import pickle
import json
import soundfile as sf
import tqdm
import time
import numpy as np
from models import audio_encoder
import tqdm
import numpy as np
from itertools import groupby
from operator import itemgetter
import csv
from collections import defaultdict
def cls_attn_seg_feats(feats, cls_attn_weights, threshold, pool, spf, level2, vad, insert_threshold, start_sec):
    # return a list of features that are segmented by cls attn weights
    threshold_value = torch.quantile(cls_attn_weights, threshold, dim=-1, keepdim=True) # [n_h, T]
    cls_attn_weights_sum = cls_attn_weights.sum(0)
    important_idx = torch.where((cls_attn_weights >= threshold_value).float().sum(0) > 0)[0].cpu().numpy()
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
    
    if level2 or len(boundaries_ex1) == 0:
        boundaries = boundaries_all
    else:
        boundaries = boundaries_ex1

    seg_feats = []
    locations = []
    boundaries_in_sec = []
    for t_s, t_e in boundaries:
        for gt_b in vad:
            if ((start_sec+t_s*spf) >= (gt_b[0] - 0.02)) and ((start_sec+t_e*spf) <= (gt_b[1] + 0.02)):
                locations.append(start_sec+spf*(t_s+t_e)/2.) # in seconds
                boundaries_in_sec.append([start_sec+t_s*spf, start_sec+t_e*spf]) # in seconds
                if pool == "mean":
                    seg_feats.append(feats[t_s:t_e].mean(0).cpu())
                elif pool == "max":
                    max_id = torch.argmax(cls_attn_weights_sum[t_s:t_e])
                    seg_feats.append(feats[t_s+max_id].cpu())
                elif pool == "median":
                    seg_feats.append(feats[int((t_s+t_e)/2)].cpu())
                elif pool == "weightedmean":
                    seg_feats.append((feats[t_s:t_e]*(cls_attn_weights_sum[t_s:t_e]/cls_attn_weights_sum[t_s:t_e].sum()).unsqueeze(1)).sum(0).cpu())
                break

    # Refinement based on VAD and insertion
    # delete all segments that completely fall into non-voiced region
    # first assign each boundaries to it's VAD region, and then draw word boundaries within that region, make sure the boundaries of the VAD is also word boundaries
    vad2bou = {}
    vad2loc = {}
    vad2feat = {}
    for gt_b in vad:
        vad2bou[f"{gt_b}"] = []
        vad2loc[f"{gt_b}"] = []
        vad2feat[f"{gt_b}"] = []
    for i in range(len(locations)):
        for gt_b in vad:
            if (locations[i] >= (gt_b[0] - 0.02)) and (locations[i] <= (gt_b[1] + 0.02)):
                vad2bou[f"{gt_b}"].append(boundaries_in_sec[i])
                vad2loc[f"{gt_b}"].append(locations[i])
                vad2feat[f"{gt_b}"].append(seg_feats[i])
                break
    for gt_b in vad:
        if len(vad2bou[f"{gt_b}"]) == 0: # in case some vad region doesn't have any attn segments
            added_s, added_e = gt_b[0], min(gt_b[0]+ 0.05, gt_b[1])
            f_s, f_e = int((added_s-start_sec) / spf), max(int((added_e-start_sec) / spf), int((added_s-start_sec) / spf)+1)
            vad2bou[f"{gt_b}"].append([added_s, added_e])
            vad2loc[f"{gt_b}"].append((added_s+added_e)/2.)
            vad2feat[f"{gt_b}"].append(feats[f_s:f_e].mean(0).cpu())
    # insert a segment in the middle when the gap between two adjacent segments are lower than threshold
    # also make sure the segment is in the voiced region
    
    interval = insert_threshold/2.
    for gt_b in vad:
        cur_boundaries_in_sec = vad2bou[f"{gt_b}"]
        for i in range(len(cur_boundaries_in_sec)):
            if i == 0:
                right_b = cur_boundaries_in_sec[i][0]
                left_b = gt_b[0]
            elif i == len(cur_boundaries_in_sec) - 1:
                right_b = gt_b[1]
                left_b = cur_boundaries_in_sec[i][1]
            else:
                right_b = cur_boundaries_in_sec[i+1][0]
                left_b = cur_boundaries_in_sec[i][1]

            gap = right_b - left_b
            if gap > insert_threshold:
                num_insert = int(gap/interval) - 1 # if two intervals can be put in the gap, then insert 1 seg to separate them, if 3 intervals can be put in the gap, then insert 2 seg to separate them...
                for insert_start in range(1, num_insert+1):
                    s_in_sec = left_b + insert_start * interval 
                    s_frame = max(int(left_b/spf), int(s_in_sec / spf))
                    e_frame = s_frame + 2
                    e_in_sec =  min(right_b, e_frame * spf)
                    vad2bou[f"{gt_b}"].append([s_in_sec, e_in_sec])
                    vad2loc[f"{gt_b}"].append((s_in_sec+e_in_sec)/2.)
                    vad2feat[f"{gt_b}"].append(feats[int(s_frame-start_sec/spf):int(e_frame-start_sec/spf)].mean(0).cpu())
        cur_locations, sorted_ind = torch.sort(torch.tensor(vad2loc[f"{gt_b}"]))
        vad2loc[f"{gt_b}"] = cur_locations.tolist()
        vad2bou[f"{gt_b}"] = np.array(vad2bou[f"{gt_b}"])[sorted_ind].tolist()
        if not isinstance(vad2bou[f"{gt_b}"][0], list):
            vad2bou[f"{gt_b}"] = [vad2bou[f"{gt_b}"]]
        vad2feat[f"{gt_b}"] = torch.stack(vad2feat[f"{gt_b}"])[sorted_ind]
    # draw word boundaries using boundaries and VAD
    # print(vad)
    # print(boundaries_in_sec)
    # print(vad2bou)
    word_boundaries = []
    for i, gt_b in enumerate(vad):
        word_boundaries_line = [gt_b[0]] # the first is vad boundary
        temp_boundaries = vad2bou[f"{gt_b}"]
        for left, right in zip(temp_boundaries[:-1], temp_boundaries[1:]):
            word_boundaries_line.append((left[1]+right[0])/2.)
        word_boundaries_line.append(gt_b[1])
        for i in range(len(word_boundaries_line)-1):
            word_boundaries.append([word_boundaries_line[i], word_boundaries_line[i+1]])
    seg_feats, locations, boundaries_in_sec = [], [], []
    for gt_b in vad:
        seg_feats.append(vad2feat[f"{gt_b}"])
        locations += vad2loc[f"{gt_b}"]
        boundaries_in_sec += vad2bou[f"{gt_b}"]
    if len(seg_feats) == 0:
        seg_feats = feats.mean(0).cpu().unsqueeze(0)
        locations = [(cls_attn_weights.shape[-1])*spf/2.]
        boundaries_in_sec = [[0.0, (cls_attn_weights.shape[-1])*spf]]
        word_boundaries = [[0.0, (cls_attn_weights.shape[-1])*spf]]
    else:
        seg_feats = torch.cat(seg_feats,dim=0)
    # print(locations)
    # print(boundaries_in_sec)
    # print(word_boundaries)
    # print(seg_feats.shape)
    assert len(word_boundaries) == len(seg_feats), f"seg_feats {len(seg_feats)}, locations {len(locations)}, boundaries_in_sec {len(boundaries_in_sec)}, word_boundaries {len(word_boundaries)}"
    return {"seg_feats": seg_feats, "locations": locations, "boundaries": boundaries_in_sec, "word_boundaries": word_boundaries}
    


print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_dir", type=str)
parser.add_argument("--dataset", type=str, default='zs20')
parser.add_argument("--data_root", type=str, default="/data2/scratch/pyp/datasets/")
parser.add_argument("--language", type=str,  default='english', choices=['english', 'french', 'LANG1', 'LANG2', 'mandarin'])
parser.add_argument("--save_root", type=str, default="/data2/scratch/pyp/discovery/word_unit_discovery/")
parser.add_argument("--feats_type", type=str, default="preFeats", choices=['preFeats', 'curFeats'])
parser.add_argument("--percentage", type=int, default=None, help="if None, the feats_type is the original name, otherwise, it's feats_type_percentage")
parser.add_argument("--threshold", type=float, default=0.90)
parser.add_argument("--reduce_method", type=str, default="mean", choices=['mean', 'max', 'median', 'weightedmean'])
parser.add_argument("--tgt_layer_for_attn", type=int, default=7, help="where attn weights are coming from, as for features, if feats_type==preFeats, and feature comes from previous layer of tgt_layer_for_attn, otherwise, feature comes from the same layer")
parser.add_argument("--level2", action="store_true", default=False, help="if True, use feats and atten weights from level2 (not avaliable for models that only has one level of w2v2)")
parser.add_argument("--segment_method", type=str, choices=['clsAttn', 'forceAlign'], default=None, help="if use cls attn segmentation or use force alignment segmentation. If use, need model_args.use_audio_cls_token to be True")
parser.add_argument("--snapshot", type=str, default='best', help='which model snapshot to use, best means best_boundle.pth, can also pass number x, say 24, then will take snapshot_24.pth')
parser.add_argument("--insert_threshold", type=float, default=10000.0, help="if the gap between two attention segments are above the threshold, we insert a two frame segment in the middle")

args = parser.parse_args()

save_root = os.path.join(args.save_root, args.exp_dir.split("/")[-1])
feats_type = args.dataset + "_" + args.feats_type + "_" + args.reduce_method + "_" + str(args.threshold) + "_" + str(args.tgt_layer_for_attn) + "_" + args.segment_method + "_" + args.language + "_" + "snapshot"+args.snapshot + "_" + "insertThreshold" + str(args.insert_threshold if args.insert_threshold < 100 else int(args.insert_threshold))

if args.percentage is not None:
    feats_type = feats_type + "_" + str(args.percentage)
save_root = os.path.join(save_root, feats_type)
print("data save at: ", save_root)
os.makedirs(save_root, exist_ok=True)
print(args)
if not os.path.isdir(args.exp_dir):
    raise RuntimeError(f"{args.exp_dir} does not exist!!")

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


data_start_time = time.time()


locF_temp = []
j = 0
# total_data = []
data_dict = {}
missing_ali = 0
level2 = False
tgt_layer = args.tgt_layer_for_attn
all_data = defaultdict(list)
######################
vad_fn = os.path.join(args.data_root,f"vads/{args.language.upper()}_VAD.pkl")
with open(vad_fn, "r") as f:
    reader = csv.reader(f, delimiter="\t")
    header = next(reader)
    for i, line in enumerate(reader):
        line = line[0].split(",")
        # if line
        all_data[line[0]].append([float(line[1]), float(line[2])])
######################


audio_root = os.path.join(args.data_root, f"2020/2017/{args.language}/train")
        
for key in tqdm.tqdm(all_data.keys()):
    pointer = 0
    wav_fn = os.path.join(audio_root, key+".wav")
    if not os.path.isfile(wav_fn):
        print(f"{wav_fn} not found")
        continue
    total_audio, sr = sf.read(wav_fn, dtype = 'float32')
    total_audio = torch.from_numpy(total_audio).unsqueeze(0).cuda()
    assert sr == 16000
    while pointer < len(all_data[key]):
        cur_vad = []
        start_sec, end_sec = all_data[key][pointer]
        cur_vad.append([start_sec, end_sec])
        while end_sec - start_sec < 4.3: # evarage duration of SC is 4.12s
            pointer += 1
            if pointer < len(all_data[key]):
                cur_vad.append(all_data[key][pointer])
                new_start_sec, end_sec = all_data[key][pointer]
            else:
                break
        if end_sec - start_sec < 0.05:
            break
        pointer += 1
        audio_use = total_audio[:, int(start_sec*sr):int(end_sec*sr+1)]
        if audio_use.shape[1]/sr < 0.05:
            print(f"VAD is longer than the actual audio for {key}")
            break
        with torch.no_grad():
            w2v2_out = model(audio_use, padding_mask=None, mask=False, need_attention_weights=True, tgt_layer=tgt_layer, pre_feats=True if feats_type == "preFeats" else False, level2=level2)

        if args.segment_method == "clsAttn": # use cls attn for segmentation
            assert model_args.use_audio_cls_token and model_args.cls_coarse_matching_weight > 0.
            feats = w2v2_out['features'].squeeze(0)[1:] # [1, T+1, D] -> [T, D]
            spf = audio_use.shape[-1]/sr/feats.shape[-2]
            attn_weights = w2v2_out['attn_weights'].squeeze(0) # [1, num_heads, tgt_len, src_len] -> [num_heads, tgt_len, src_len]
            cls_attn_weights = attn_weights[:, 0, 1:] # [num_heads, tgt_len, src_len] -> [n_h, T]
            out = cls_attn_seg_feats(feats, cls_attn_weights, args.threshold, args.reduce_method, spf, level2, cur_vad, args.insert_threshold, start_sec)
        else:
            raise NotImplementedError(f"doesn't support {args.segment_method}")
        
        seg_feats = out['seg_feats'].cpu()
        
        data_dict[f"{key}_{start_sec:.2f}-{end_sec:.2f}"] = {"seg_feats": seg_feats, "locations": torch.tensor(out['locations']), "boundaries": torch.tensor(out['word_boundaries']), "spf":spf}

with open(os.path.join(save_root, 'data_dict.pkl'), "wb") as f:
    pickle.dump(data_dict, f)
print(f"save pickle data at {os.path.join(save_root, 'data_dict.pkl')}")


