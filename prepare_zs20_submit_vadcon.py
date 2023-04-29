import argparse
import json
import os
import time
import tqdm
from collections import defaultdict
print("\nI am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_json", type=str, default="/data1/scratch/coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy_with_alignments.json", help="the fn of force alignment json file")
parser.add_argument("--save_root", type=str, default="/data2/scratch/pyp/exp_pyp")
parser.add_argument("--language", type=str,  default='english', choices=['english', 'french', 'LANG1', 'LANG2', 'mandarin'])
parser.add_argument("--exp_dir", type=str, default="/scratch/cluster/pyp/exp_pyp/discovery/word_unit_discovery/disc-23/curFeats_mean_0.9_7_forceAlign")
parser.add_argument("--k", type=int, default=4096)
# parser.add_argument("--code_halfwidth", type=float, default=0.01, choices=[0.01, 0.04], help="0.01 is for level1 w2v2, 0.04 is for level2 w2v2") # not used, the save data provides this
parser.add_argument("--run_length_encoding", action="store_true", default=False, help="if True, collapse all adjacent same code into one code; if False, use the original implementation, which, when calculate word2code_recall, it collapse all same code within the same word into one code. and when calculating code2word_precision, it doesn't do anything, so if a code appears 10 times (within the interval of a word), this are accounted as coappearing 10 times ")
parser.add_argument("--iou", action="store_true", default=False, help="wether or not evaluate the intersection over union, center of mass distance, center of mass being in segment percentage")
parser.add_argument("--max_n_utts", type=int, default=200000, help="total number of utterances to study, there are 25020 for SpokenCOCO, so if the number is bigger than that, means use all utterances")
parser.add_argument("--topk", type=int, default=30, help="show stats of the topk words in hisst plot")
parser.add_argument("--tolerance", type=float, default=0.02, help="tolerance of word boundary match")
parser.add_argument("--snapshot", type=str, default='best', help='which model snapshot to use, best means best_boundle.pth, can also pass number x, say 24, then will take snapshot_24.pth')
parser.add_argument("--insert_threshold", type=float, default=10000.0, help="if the gap between two attention segments are above the threshold, we insert a two frame segment in the middle")


args = parser.parse_args()


def cal_code_boundary(data, spk):
    boundaries = data['boundaries']
    codes = data['codes']
    cur_code2seg = defaultdict(list)
    for i, code in enumerate(codes):
        s = round(boundaries[i][0],2)
        e = round(boundaries[i][1],2)
        if s < e:
            cur_code2seg[f"Class {code}"].append(f"{spk} {s:.2f} {e:.2f}")
    return cur_code2seg

# submission file 
# name: in the dir
# out fn: /data1/scratch/exp_pyp/zs2020/2017/english.txt
# in english.txt the file should look like 
# Class 0
# s0019 5839.17 5839.43
# s0107 3052.89 3053.17
# s0107 4657.09 4657.45
# s1724 5211.24 5211.59
# s1724 10852.39 10852.72
# s2544 4561.61 4561.9
# s2544 6186.02 6186.36
# s2544 8711.48 8711.73
# s3020 11256.47 11256.82
# s5157 459.55 459.86
# s5968 1359.01 1359.3

# Class 1
# s0107 6531.34 6531.63
# s4018 206.01 206.31
# s6519 547.35 547.69
#

def prepare_data(exp_dir):
    with open(os.path.join(exp_dir, "code_dict.json"), "r") as f:
        code_dict = json.load(f)
    code2seg = defaultdict(list)
    for key in tqdm.tqdm(code_dict):
        spk = "_".join(key.split("_")[:-1])
        cur_code2seg = cal_code_boundary(code_dict[key], spk) # {f"Class {code}": [f"spk str(start_sec) str(end_sec)", ...]]}
        for class_name in cur_code2seg:
            code2seg[class_name] += cur_code2seg[class_name]
    return code2seg


code2seg = prepare_data(args.exp_dir)

out_dir = os.path.join(args.save_root, "zs2020_snapshot" + args.snapshot + "_insertThreshold" + str(args.insert_threshold if args.insert_threshold < 100 else int(args.insert_threshold)) + "/2017/track2")
if not os.path.isdir(out_dir):
    os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, f"{args.language}.txt"), "w") as f:
    i = 0
    for j, key in enumerate(code2seg):
        if len(code2seg[key]) == 0:
            continue
        f.write(key+"\n")
        for item in code2seg[key]:
            f.write(f"{item}\n")
        i += 1
        # if j < len(res)-1:
        #     f.write("\n")
        f.write("\n")
print(f"find {i} classes in total")