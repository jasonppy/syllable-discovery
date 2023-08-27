# Syllable Segmentation and Cross-Lingual Generalization in a Visually Grounded, Self-Supervised Speech Model


# Table of Contents
1. [Environment](#1-environment)
2. [Apply VG-HuBERT to Syllable Level Speech Segmentation](#2-apply-vg-huBERT-on-syllable-level-speech-segmentation)
3. [Speech Segmentation and Syllable Detection on SpokenCOCO](#3-speech-segmentation-and-syllable-detection-on-spokencoco)
4. [Apply VG-HuBERT to ZeroSpeech2020](#4-apply-vg-hubert-on-zerospeech2020)
5. [Apply VG-HuBERT to the Estonian Conversational Corpus](#5-apply-vg-hubert-to-the-estonian-conversational-corpus)
6. [Training](#6-training)

## 1. Environment
It is recommended to create a new conda environment for this project with `conda create -n sd python=3.9`, the requirement on python version is not rigid, as long as you can install the packages listed in `./requirements.txt`. The requirement for the versions of the packages is not rigid either, while the listed versions were tested, higher/lower versions might also work.

If you want to get the attention weights of different attention head (**which is required for all word and boundary detection experiments**), you need to modify the output of the `multi_head_attention_forward` function in the PyTorch package at`torch/nn/functional`. if you install pytorch using conda in environment `sd`, the path of the file should be `path_to_conda/envs/sd/lib/python3.9/site-packages/torch/nn/functional.py`. get to function `multi_head_attention_forward`, and change the output as the following

```python
    # if need_weights:
    #     # average attention weights over heads
    #     attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    #     return attn_output, attn_output_weights.sum(dim=1) / num_heads
    # else:
    #     return attn_output, None
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    return attn_output, attn_output_weights
```

Simply put, originally, the return of `attn_output_weights` is summed over all attention heads, and we don't want to do that so that we can have the attention weights from different heads.

Note that since PyTorch 1.11, `multi_head_attention_forward` accepts argument `average_weights` which controls whether returning averaged attention or unaveraged attention. However, for minimal code change, we recommend ignore this argument and change the code in `multi_head_attention_forward` as

```python
    # if need_weights:
    #     # optionally average attention weights over heads
    #     attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    #     if average_attn_weights:
    #         attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

    #     if not is_batched:
    #         # squeeze the output if input was unbatched
    #         attn_output = attn_output.squeeze(1)
    #         attn_output_weights = attn_output_weights.squeeze(0)
    #     return attn_output, attn_output_weights
    # else:
    #     if not is_batched:
    #         # squeeze the output if input was unbatched
    #         attn_output = attn_output.squeeze(1)
    #     return attn_output, None
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    return attn_output, attn_output_weights
```

## 2. Apply VG-HuBERT on Syllable Level Speech Segmentation
To enable quickly applying the VG-HuBERT on speech segmentation, we provide the following standalone script. You need to provide four arguments to make it run:

1. `model_path`. It should be the directory the `.pth` and `args.pkl` are at. We open provide two checkpoints of VG-HuBERT [VG-HuBERT_3](https://www.cs.utexas.edu/~harwath/model_checkpoints/vg_hubert/vg-hubert_3.tar) best_bundle.pth is better are syllable segmentation, snapshot_20.pth is better at word segmentation. See the paper for detailed results and discussion on the reasons.

2. `wav_file`. The speech file you want to segment, we recommend the length of the speech to be 1 ~ 8 seconds, although in our experience the segmentation performance of VG-HuBERT is robust to the length of the input. the file should be [SoundFlie](https://pysoundfile.readthedocs.io/en/latest/) Readable, i.e. .wav, .flac etc.

3. `tgt_layer`, `segment_method` and `secPerSyllable`. `tgt_layer` is the layer from which you want to get the feature self-similarity matrix from, `segment_method` can be `minCut` or `minCutMerge-${mergeThres}`, we recommend setting mergeThres to be 0.3, 0.35, or 0.4. And 0.3 works the best for English. Consider secPerSyllable to be 0.15 or 0.2 (0.2 works the best for English)

```python
model_path = # TODO
wav_file = # TODO
tgt_layer = # TODO
segment_method = # TODO
secPerSyllable = # TODO



import torch
import soundfile as sf
from models import audio_encoder
import numpy as np
from mincut import mincut

def mincut_wrapper(audio_len_sec, feat, spf, attn_weights, segment_method):
    num_syllable = int(np.ceil(audio_len_sec / args.secPerSyllable))

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

    boundaries = [[l*spf,r*spf] for l, r in seg_boundary_frame_pairs]
    if args.reduce_method == "mean":
        feat = [torch.from_numpy(feat[round(l):round(r)].mean(0)) for l,r in seg_boundary_frame_pairs]
    elif args.reduce_method == "median":
        feat = [torch.from_numpy(feat[min(round(l+r/2), len(feat)-1)]) for l,r in seg_boundary_frame_pairs]
    elif args.reduce_method == "max":
        feat = [torch.from_numpy(feat[l+np.argmax(attn_weights[l:r])]) for l,r in seg_boundary_frame_pairs]
    else:
        raise NotImplementedError

    return feat, boundaries

# setup model
with open(os.path.join(model_path, "args.pkl"), "rb") as f:
    model_args = pickle.load(f)
model = audio_encoder.AudioEncoder(model_args)
bundle = torch.load(os.path.join(model_path, "best_bundle.pth"))
model.carefully_load_state_dict(bundle['dual_encoder'], load_all=True)
model.eval()
model = model.cuda()

# load waveform (do not layer normalize the waveform!)
audio, sr = sf.read(wav_file, dtype = 'float32')
assert sr == 16000
audio_len_in_sec = len(audio) / sr
audio = torch.from_numpy(audio).unsqueeze(0).cuda() # [T] -> [1, T]

# model forward
out = model(torch.from_numpy(audio).unsqueeze(0).cuda(), padding_mask=None, mask=False, tgt_layer=args.layer, need_attention_weights=True, pre_feats= False)
feat = out['features'].squeeze(0)[1:].cpu().float().numpy()
attn_weights = out['attn_weights'].squeeze(0)[:, 0, 1:] # [num_heads, tgt_len, src_len] -> [n_h, T]
attn_weights = attn_weights.sum(0).cpu().float().numpy() # [T]

spf = audio.shape[0]/sr/feat.shape[0]

pooled_feat, boundaries = mincut_wrapper(audio_len_sec=audio.shape[0]/sr, feat=feat, spf=spf, attn_weights=attn_weights, segment_method=segment_method)

```
## 3. Speech Segmentation and Syllable Detection on SpokenCOCO
This section illustrates how to apply the VG-HuBERT model to segment speech and detect words in SpokenCOCO. Please first download the SpokenCOCO audios and MSCOCO images following:
```bash
coco_root=/path/to/coco/
wget https://data.csail.mit.edu/placesaudio/SpokenCOCO.tar.gz -P ${coco_root} # 64G
wget http://images.cocodataset.org/zips/train2014.zip -P {coco_root}
wget http://images.cocodataset.org/zips/val2014.zip -P {coco_root}
```
Please untar/unzip the compressed files after downloading them

Then download karpathy split json files with syllable alignment from the following link:
[val](https://drive.google.com/file/d/1ZqszndqeiV8W8pV_7YVWIngmaza3cf9I/view?usp=drive_link), [test](https://drive.google.com/file/d/1e8dgwDWcHpB1Bf_q_T6v2JiKTD33f9Ts/view?usp=drive_link)

Then you are all set, just run

```bash
# to get boundary results only, run the follow
cd ./scripts
mkdir -p logs
sps=0.2
layer=8
mergeThres=0.3
bash spokencoco_boundary.sh vg-hubert_3 ${sps} ${layer} mean minCutMerge-${mergeThres} 1 best 32

# to get the full results, including clustering, run the following
cd ./scripts
mkdir -p logs
sps=0.2
layer=8
mergeThres=0.3
bash spokencoco_clustering.sh vg-hubert_3 ${sps} ${layer} 16384 mean minCutMerge-${mergeThres} 1 best 32 1.1 0 classic 4096 cosine average -0.02
```

The output of the first command gives:
```log
best_shift: -0.020000000000000018
boundary precision: 0.5735236764063061
boundary recall:  0.6357311344704828
boundary F1: 0.6030273424061868
boundary over-segmentation:  0.10846537052134986
boundary R value:  0.6428260122957377
```
The result is deterministic


The output of the second command gives:
```log
any instance in data_dict that is not in data_json: {}
Took 13s to dump 360017 codes for 22031 utts
There are 4096 code clusters
AScore: 0.7479
IoU: 0.6275
IoT: 0.7792
Percentage that the segment falls within the word interval: 27.69
Average distance (in seconds) between segment center and word center: 0.0449
Percentage that word centers fall into the code segment: 93.44%
code coverage (average over all *words*): 0.9253
code coverage (average over all *word types*): 0.9679
coverage per sentence: 0.9124
boundary precision: 0.5737
boundary recall: 0.6359
boundary F1: 0.6032
boundary over-segmentation: 0.1085
boundary R value: 0.6430
purity: 0.4582
902 / 2498 words with an F1 score >= 0.5
889 / 3519 codes with an F1 score >= 0.5
2002 / 3519 codes with an precision score >= 0.5
1041 / 3519 codes with an recall score >= 0.5
avg F1 = 94.70% for top 250 words; 38.91% for all 2482 words
good words/total words: 55216/137772 -> 0.40077809714600937
good codes/total codes: 55580/203106 -> 0.27365021220446467
..."a big table on discovered syllables"...
```
Since we use FAISS batched kmeans, and sklearn agglomerative cluster, there are minor randomness, the boundary and area numbers are deterministic as before, but clustering related results should vary by 2~3% with different machines and random seeds


# 4 Apply VG-HuBERT to ZeroSpeech2020
First follow the ZeroSpeech 2020 section on the ZeroSpeech website to download the data and ground truth labels (remember to also download and unzip `2017_vads.zip`). This should be free and easy, like the rest of the steps :). Suppose you have put the `2020` folder at `/zs20/`. 

Then install zerospeech 2020 evaluation toolkit following [this official repo](https://github.com/zerospeech/zerospeech2020). Assume you have clone the repo at `~/zerospeech2020`.

Now you should be ready to test the models on this task. similarly, change the `model_root` and `data_root` in `./scripts/run_zs20.sh` to the parent folder of your model folder and data folder (for data_root, is should be `/zs20` if you follow the above)

Then run
```bash
cd ./scripts
bash zs20.sh vg-hubert_3 9 16384 max clsAttn 1 20 mandarin 10000
bash zs20.sh vg-hubert_3 9 16384 max clsAttn 1 20 english 10000
bash zs20.sh vg-hubert_3 9 16384 max clsAttn 1 20 french 10000
bash zs20.sh vg-hubert_3 9 16384 max clsAttn 1 20 LANG1 10000
bash zs20.sh vg-hubert_3 9 16384 max clsAttn 1 20 LANG2 10000
```

We can only have results for Mandarin, English and French, as the other two needs to get by submitting the prediction to the official challenge website

Results:
```json
{
    "2017-track2": {
        "mandarin": {
            "scores": {
                "ned": 0.7021258523047275,
                "coverage": 1.0,
                "words": 12591
            },
            "details": {
                "boundary_precision": 0.5112769366913014,
                "boundary_recall": 0.8604145703360837,
                "boundary_fscore": 0.6414126260201632,
                "grouping_precision": 0.13535751535519844,
                "grouping_recall": 0.1363928300344477,
                "grouping_fscore": 0.1358732005234841,
                "token_precision": 0.17021116138763198,
                "type_precision": 0.14820109602096737,
                "token_recall": 0.2280258638108709,
                "type_recall": 0.21034832600608724,
                "token_fscore": 0.1949218412643579,
                "type_fscore": 0.173888733575622,
                "words": 12591,
                "coverage": 1.0,
                "ned": 0.7021258523047275,
                "pairs": 17051
            }
        }
    }
}


{
    "2017-track2": {
        "english": {
            "scores": {
                "ned": 0.3913583129473666,
                "coverage": 0.9991355816758402,
                "words": 87552
            },
            "details": {
                "boundary_precision": 0.5695428258401303,
                "boundary_recall": 0.6786049022198433,
                "boundary_fscore": 0.6193089887524657,
                "grouping_precision": "NA",
                "grouping_recall": "NA",
                "grouping_fscore": "NA",
                "token_precision": 0.2623088570078605,
                "type_precision": 0.08831323099415204,
                "token_recall": 0.2705930898496782,
                "type_recall": 0.3676302776721187,
                "token_fscore": 0.2663865821142113,
                "type_fscore": 0.14241508877919398,
                "words": 87552,
                "coverage": 0.9991355816758402,
                "ned": 0.3913583129473666,
                "pairs": 6694523
            }
        }
    }
}


{
    "2017-track2": {
        "french": {
            "scores": {
                "ned": 0.5952500126237648,
                "coverage": 0.9996943971952165,
                "words": 61661
            },
            "details": {
                "boundary_precision": 0.4778421844809358,
                "boundary_recall": 0.6665964348896938,
                "boundary_fscore": 0.5566535264076737,
                "grouping_precision": 0.44937768743958295,
                "grouping_recall": 0.5448100923979342,
                "grouping_fscore": 0.49251359628568697,
                "token_precision": 0.14581803473746627,
                "type_precision": 0.0612704951265792,
                "token_recall": 0.16616399369943438,
                "type_recall": 0.17383702204021534,
                "token_fscore": 0.1553275816993464,
                "type_fscore": 0.09060603880375087,
                "words": 61661,
                "coverage": 0.9996943971952165,
                "ned": 0.5952500126237648,
                "pairs": 2904453
            }
        }
    }
}

```

## 5. Apply VG-HuBERT to the Estonian Conversational Corpus
Please first obtain a signed agreement following [the official corpus website](https://datadoi.ee/bitstream/handle/33/351/ekskfk_info_eng.html?sequence=45)

After that please download the data following instructions on the same website, and unzip and put the data in `/path/to/wav`. Contact me with the agreement and I'll send you the validation split, and preprocessed testing data. Please put the valid split at `/path/to/valid_pkl` and the testing data at `/path/to/test/data`. Change the corresponding path in `scripts/estonian_boundary.sh`

To get the testing rest in the paper, run
```bash
cd ./scripts
sps=0.15
layer=8
mergeThres=0.4
bash estonian_boundary.sh vg-hubert_3 ${sps} ${layer} mean minCutMerge-${mergeThres} 1 best 32 estonian
```

results are the following
```log
shift: -0.01500000000000002
prec: 0.7748138039886228
rec:  0.7992825766184427
os: 0.03158019707942539
f1:  0.7868580110866227
R val:  0.816277090533229
```

## 6. Training
To train a VG-HuBERT model, check out the [codebase](https://github.com/jasonppy/word-discovery) for [the word discovery with VG-HuBERT paper](https://arxiv.org/pdf/2203.15081.pdf)
