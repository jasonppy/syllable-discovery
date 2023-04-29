import argparse
import torch
import os
import pickle
import json
import soundfile as sf
import tqdm
import time
import numpy as np

from mincut import mincut
from models import audio_encoder


def get_word_ali(raw_ali):
    """
    raw_ali is a string like 'start1__word1__end1 start2__word2__end2 ...'
    """
    data = []
    meta_toks = raw_ali.split()
    for meta_tok in meta_toks:
        toks = meta_tok.split('__')
        if len(toks) == 3:
            data.append((float(toks[0]), float(toks[2]), toks[1]))
    return data


print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_dir", type=str)
parser.add_argument("--dataset", type=str, default='spokencoco')
parser.add_argument("--seg_fn", type=str, default='no', help="path to the segmentation (i.e. boundaries) file, if not provided, use do segmentation and feature extraction on the fly")
parser.add_argument("--snapshot", type=str, default='best', help='which model snapshot to use, best means best_boundle.pth, can also pass number x, say 24, then will take snapshot_24.pth')
parser.add_argument("--data_json", type=str, default="/data1/scratch/coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy_with_alignments.json")
parser.add_argument("--audio_base_path", type=str, default="/data1/scratch/coco_pyp/SpokenCOCO")
parser.add_argument("--max_iter", type=int, default=200)
parser.add_argument("--save_root", type=str, default="/data2/scratch/pyp/discovery/word_unit_discovery/")
parser.add_argument("--reduce_method", type=str, default="mean", choices=['mean', 'max', 'median', 'weightedmean'])
parser.add_argument("--layer", type=int, default=7, help="where attn weights are coming from, as for features, if feats_type==preFeats, and feature comes from previous layer of tgt_layer_for_attn, otherwise, feature comes from the same layer")
parser.add_argument("--level2", action="store_true", default=False, help="if True, use feats and atten weights from level2 (not avaliable for models that only has one level of w2v2)")
parser.add_argument("--segment_method", type=str, default=None, help="example choices=['minCut', 'forceAlign', 'uniform', 'minCutMerge-0.5']")
parser.add_argument("--spf", type=float, default=0.02)
parser.add_argument("--secPerSyllable", type=float, help="second per segment, given audio_len_sec, we get K in the min cut algorithm by audio_len_sec/secPerSyllable, 0 means use the ground truth")
parser.add_argument("--n_proc", type=int, default=32, help="number of parallel process for min cut segmentation")
args = parser.parse_args()

feats_type = args.dataset + "_" + args.segment_method + "_" + "secPerSyllable" + str(args.secPerSyllable if args.secPerSyllable > 0 else int(args.secPerSyllable)) + "_" + str(args.layer) + "_" + args.reduce_method + "_" + "snapshot" + args.snapshot

save_root = os.path.join(args.save_root, feats_type)
print("data save at: ", save_root)
os.makedirs(save_root, exist_ok=True)
print(args)
if not os.path.isdir(args.exp_dir) and "vg" in args.exp_dir.lower():
    raise RuntimeError(f"{args.exp_dir} does not exist!!")




data_start_time = time.time()

with open(args.data_json, "r") as f:
    data_json = json.load(f)
    if "data" in data_json:
        data_json = data_json['data']

############# get segment boundaries if specified #############
if args.seg_fn != "no":
    assert os.path.isfile(args.seg_fn)
    with open(args.seg_fn, "rb") as f:
        segment_dict = pickle.load(f)
############# get segment boundaries if specified #############

locF_temp = []
j = 0
# total_data = []
temp_data_dict = {}
missing_ali = 0
temp_fn = os.path.join(args.save_root, f'{args.dataset}_{args.snapshot}_{args.layer}_data_dict.pkl')
if os.path.isfile(temp_fn):
    print(temp_fn, "exists, load the existing one" )
    with open(temp_fn, "rb") as f:
        temp_data_dict = pickle.load(f)
    print(f"len of the temp_data_dict: {len(temp_data_dict)}")
else:
    if 'hubert' in args.exp_dir.lower() and 'vg' not in args.exp_dir.lower():
        from transformers import Wav2Vec2Processor, HubertModel
        model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        model = model.cuda()
        model = model.eval()
        model_args = None
    else:
        ########################## setup model ##########################
        with open(os.path.join(args.exp_dir, "args.pkl"), "rb") as f:
            model_args = pickle.load(f)
        print(args.exp_dir)
        print(args.exp_dir)
        print(args.exp_dir)
        print(args.exp_dir)
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
    for j, item in enumerate(tqdm.tqdm(data_json)):
        raw_ali = item.get('syllable_alignment', None)
        if raw_ali is None:
            continue
        if 'caption' in item:
            wav_fn = item['caption']['wav']
        else:
            assert "librispeech" in args.dataset
            wav_fn = item['wav']
        key = os.path.join(args.audio_base_path, wav_fn)

        
        audio, sr = sf.read(key, dtype = 'float32')
        assert sr == 16000
        assert len(audio.shape) == 1, audio.shape
        with torch.no_grad():
            if "hubert" in args.exp_dir.lower() and "vg" not in args.exp_dir.lower():
                out = model(input_values=torch.from_numpy(audio).unsqueeze(0).cuda(), output_hidden_states=True, return_dict=True)
                feat = out['hidden_states'][args.layer][0].cpu().float().numpy()
                attn_weights = np.random.randn(feat.shape[-1])
            else:
                out = model(torch.from_numpy(audio).unsqueeze(0).cuda(), padding_mask=None, mask=False, tgt_layer=args.layer, need_attention_weights=True, pre_feats= False)
                if model_args != None and not model_args.use_audio_cls_token: # to handle SC finetuned HuBERT (disc-63 on RTX/a40)
                    feat = out['features'].squeeze(0).cpu().float().numpy()
                    attn_weights = np.random.randn(12, len(feat))
                else:
                    feat = out['features'].squeeze(0)[1:].cpu().float().numpy()
                    attn_weights = out['attn_weights'].squeeze(0)[:, 0, 1:] # [num_heads, tgt_len, src_len] -> [n_h, T]
                    attn_weights = attn_weights.sum(0).cpu().float().numpy() # [T]

        spf = audio.shape[0]/sr/feat.shape[0]
        
        ali = get_word_ali(raw_ali)
        temp_data_dict[wav_fn] = {"feat": feat, "attn_weight": attn_weights, "audio_len_sec": (audio.shape[0]/sr), "spf": spf, "ali": ali}
        # if j >= 1000:
        #     break
    print("save the extracted temp_data_dict at ", temp_fn)
    with open(temp_fn, "wb") as f:
        pickle.dump(temp_data_dict, f)

data_dict = {}

def mincut_parallel(audio_len_sec, ali, feat, spf, wav_fn, attn_weight, segment_method):
    if args.secPerSyllable >= 10000: # probably the best guess of num_sullables, actually n_syl + n_sil blocks
        n_sil = [True for l,r in zip(ali[:-1], ali[1:]) if r[0] - l[1] >= 0.02] # if two syllables are further than 0.02, then there is a silence block
        n_sil.append(True if ali[0][0] >= 0.02 else False)
        n_sil = sum(n_sil)
        num_syllable = len(ali) + n_sil
    else:
        num_syllable = int(np.ceil(audio_len_sec / args.secPerSyllable)) if args.secPerSyllable > 0 else len(ali) - int(args.secPerSyllable) # if negative number, means number of ground truth - that number

    ssm = feat@feat.transpose(1,0)
    ssm = ssm - np.min(ssm) + 1e-7 # make it non-negative
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
    if args.reduce_method == "mean":
        feat = [torch.from_numpy(feat[round(l):round(r)].mean(0)) for l,r in seg_boundary_frame_pairs]
    elif args.reduce_method == "median":
        feat = [torch.from_numpy(feat[min(round(l+r/2), len(feat)-1)]) for l,r in seg_boundary_frame_pairs]
    elif args.reduce_method == "max":
        feat = [torch.from_numpy(feat[l+np.argmax(attn_weight[l:r])]) for l,r in seg_boundary_frame_pairs]
    else:
        raise NotImplementedError
    boundaries = word_boundaries
    locations = [(l+r)/2 for l,r in word_boundaries]
    return feat, locations, boundaries, word_boundaries, spf, wav_fn

if "minCut" in args.segment_method and args.n_proc > 1:
    # parallel processing for faster results
    import joblib
    parallizer = joblib.Parallel(n_jobs=args.n_proc, max_nbytes=None, verbose=2)
    res = parallizer(joblib.delayed(mincut_parallel)(temp_data_dict[wav_fn]['audio_len_sec'], temp_data_dict[wav_fn]['ali'], temp_data_dict[wav_fn]['feat'], temp_data_dict[wav_fn]['spf'], wav_fn, temp_data_dict[wav_fn]['attn_weight'], args.segment_method) for wav_fn in temp_data_dict.keys())
    for item in res:
        data_dict[item[-1]] = {"seg_feats": torch.stack(item[0], dim=0), "locations": torch.tensor(item[1]), "boundaries": torch.tensor(item[2]), "word_boundaries": item[3], "spf":item[4]}

else:
    # sequential process for debugging:
    for wav_fn in tqdm.tqdm(temp_data_dict):
        spf = temp_data_dict[wav_fn]['spf']
        if args.segment_method == "forceAlign":
            
            word_boundaries = [[item[0], item[1]] for item in temp_data_dict[wav_fn]['ali']]
            if args.reduce_method == "mean":
                feat = [torch.from_numpy(temp_data_dict[wav_fn]['feat'][round(ind[0]/spf):round(ind[1]/spf+1)].mean(0)) for ind in word_boundaries]
            elif args.reduce_method == "median":
                feat = [torch.from_numpy(temp_data_dict[wav_fn]['feat'][min(round((ind[0]/spf+(ind[1]/spf+1))/2), len(feat)-1)]) for ind in word_boundaries]
            else:
                raise NotImplementedError
        elif args.segment_method == "uniform":
            num_syllable = int(np.ceil(temp_data_dict[wav_fn]['audio_len_sec'] / args.secPerSyllable)) if args.secPerSyllable > 0 else len(temp_data_dict[wav_fn]['ali'])
            if num_syllable <= 0:
                print(wav_fn, "has only one syllable as predicted, but it's raw alignment is ", raw_ali)
                continue
            mul = len(feat) // num_syllable
            seg_boundary_frame = list(range(len(temp_data_dict[wav_fn]['feat']))[::mul])
            word_boundaries = [[l*spf,r*spf] for l, r in zip(seg_boundary_frame[:-1], seg_boundary_frame[1:])]
            # feat = [torch.from_numpy(temp_data_dict[wav_fn]['feat'][round(l):round(r)].mean(0)) for l,r in zip(seg_boundary_frame[:-1], seg_boundary_frame[1:])]
            if args.reduce_method == "mean":
                feat = [torch.from_numpy(feat[round(l):round(r)].mean(0)) for l,r in zip(seg_boundary_frame[:-1], seg_boundary_frame[1:])]
            elif args.reduce_method == "median":
                feat = [torch.from_numpy(feat[min(round(l+r/2), len(feat)-1)]) for l,r in zip(seg_boundary_frame[:-1], seg_boundary_frame[1:])]
            else:
                raise NotImplementedError
        elif args.segment_method == "minCut":
            num_syllable = int(np.ceil(temp_data_dict[wav_fn]['audio_len_sec'] / args.secPerSyllable)) if args.secPerSyllable > 0 else len(temp_data_dict[wav_fn]['ali'])
            if num_syllable <= 0:
                print(wav_fn, "has only one syllable as predicted, but it's raw alignment is ", raw_ali)
                continue
            seg_boundary_frame = mincut.min_cut(temp_data_dict[wav_fn]['feat']@temp_data_dict[wav_fn]['feat'].transpose(1,0), num_syllable+1) # +1 for the algo
            word_boundaries = [[l*spf,r*spf] for l, r in zip(seg_boundary_frame[:-1], seg_boundary_frame[1:])]
            feat = [torch.from_numpy(temp_data_dict[wav_fn]['feat'][round(l):round(r)].mean(0)) for l,r in zip(seg_boundary_frame[:-1], seg_boundary_frame[1:])]
        else:
            raise NotImplementedError(f"segmentation method: {args.segment_method} is not implemented")
        boundaries = word_boundaries
        locations = [(l+r)/2 for l,r in word_boundaries]
        
        data_dict[wav_fn] = {"seg_feats": torch.stack(feat, dim=0), "locations": torch.tensor(locations), "boundaries": torch.tensor(boundaries), "word_boundaries": word_boundaries, "spf":spf}

with open(os.path.join(save_root, 'data_dict.pkl'), "wb") as f:
    pickle.dump(data_dict, f)
print(f"save pickle data at {os.path.join(save_root, 'data_dict.pkl')}")


