import pickle
import numpy as np
import faiss
import tqdm
import soundfile as sf
import torch
import joblib
import scipy.spatial.distance
import os


import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--wav_root", type=str, default="/data/scratch/pyp/datasets/coco_pyp/SpokenCOCO/")
parser.add_argument("--data_dict_root", type=str, default="/saltpool0/scratch/pyp/more_discovery/word_unit_discovery/")
parser.add_argument("--data_dict_fn", type=str, default="disc-81/spokencoco_preFeats_mean_0.95_10_clsAttn_vadno_insertThreshold10000_snapshotbest/data_dict.pkl")
parser.add_argument("--n_parallize", type=int, default=8)
parser.add_argument("--kmeans_k", type=int, default=4096)
parser.add_argument("--distance_quantile", type=float, default=0.9, help="only retain entries whose distance is smaller than top 90%, this is for kmeans clustering")
parser.add_argument("--cluster_drop", type=int, default=0, help="when a cluster has lower than cluster_drop instance that are smaller than distance threshold, drop this cluster")
parser.add_argument("--agglomerative_approach", type=str, default='classic', choices=['classic', 'customized'], help="if classic, use sklearn, if customized, that means we'll consider that each instances is actually a kmeans cluster that contains many instances, and then if linkage is average, each the feature of merged clusters will be the weighted average based on actually data point is constains")
parser.add_argument("--agglomerative_k", type=int, default=2048)
parser.add_argument("--affinity", type=str, choices=['euclidean', 'l1', 'l2', 'manhattan', 'cosine'], help='If linkage is “ward”, only “euclidean” is accepted')
parser.add_argument("--linkage", type=str, choices=['ward', 'complete', 'average', 'single'])
args = parser.parse_args()

if args.debug:
    args.vis_size = 10000

data_dict_fn = os.path.join(args.data_dict_root, args.data_dict_fn)
save_root = os.path.join(os.path.dirname(data_dict_fn), f"kmeansK{args.kmeans_k}_distanceQuantile{args.distance_quantile}_clusterDrop{args.cluster_drop}_aggApproach{args.agglomerative_approach}_aggK{args.agglomerative_k}_affinity{args.affinity}_linkage{args.linkage}")
kmeans_save_fn = os.path.join(os.path.dirname(data_dict_fn), str(args.kmeans_k),"centroids.npy")

os.makedirs(save_root, exist_ok=True)

## load features, should be a pkl with the following field
# feats_dcit = {"seg_feats": out['seg_feats'], "locations": torch.tensor(out['locations']), "boundaries": torch.tensor(out['boundaries']), "word_boundaries": out['word_boundaries'], "spf":spf}
# train the kmeans model on seg_feats of feats_dcit
# use kmeans to assign code to each segments, add "code" field to dict feats_dcit
print("....load features....")
with open(data_dict_fn, "rb") as f:
    feats_dict = pickle.load(f)
feats = []
for key in feats_dict:
    feats.append(feats_dict[key]['seg_feats'].numpy())
feats = np.concatenate(feats)

print("....train kmeans....")
kmeans = faiss.Kmeans(
        feats.shape[1],
        500 if args.debug else args.kmeans_k,
        niter=5 if args.debug else 50, # might need to increase it
        verbose=True,
        spherical=False,
        max_points_per_centroid=feats.shape[0],
        gpu=True,
        nredo=3 if args.debug else 5, # might need to increase it
        seed = args.seed
    )
kmeans.train(feats)

## save kmeans model
os.makedirs(os.path.dirname(kmeans_save_fn), exist_ok=True)
np.save(kmeans_save_fn, kmeans.centroids)

## cluster embed
print("....assign kmeans clusters....")
code_dict = {}
for j, key in enumerate(tqdm.tqdm(feats_dict, disable=True)):
    cluster_distances, cluster_indices = kmeans.assign(feats_dict[key]['seg_feats'].numpy())
    cur_code = cluster_indices.tolist()
    feats_dict[key].update({"code": cur_code})

    ## extract the above dict into code_dict = {code: "utt": [...], "boundaries": [...], "index": [12,24,...]}
    # index is the index of the corresponding boundary in the utt, which can be use to extract the original dict
    for i, (clus_dist, code, boundary, seg_feats) in enumerate(zip(cluster_distances, cur_code, feats_dict[key]['boundaries'], feats_dict[key]['seg_feats'])):
        if code not in code_dict:
            code_dict[code] = {"utt":[], "kmeans_distance":[], "boundaries": [], "index": [], "vghubert_feat": []}
        code_dict[code]['utt'].append(key)
        code_dict[code]['kmeans_distance'].append(clus_dist)
        code_dict[code]['boundaries'].append(boundary)
        code_dict[code]['index'].append(i)
        code_dict[code]['vghubert_feat'].append(seg_feats.numpy())
    if args.debug:
        if j > 200:
            break

for code in code_dict:
    code_dict[code]['vghubert_feat'] = np.stack(code_dict[code]['vghubert_feat'], axis=0)

print("....filter garbage instances and garbage cluster....")
all_distance = [d for item in code_dict.values() for d in item["kmeans_distance"] ]

if args.distance_quantile < 1:
    distance_threshold = np.quantile(all_distance, args.distance_quantile)
    if args.cluster_drop > 0:
        for code in code_dict.copy():
            if 'kmeans_distance' in code_dict[code] and np.sum(code_dict[code]['kmeans_distance'] <= distance_threshold) < args.cluster_drop:
                code_dict[code]['kmeans_distance'] = None
    new_code_dict = {}
    for code in code_dict:
        if 'kmeans_distance' in code_dict[code] and code_dict[code]['kmeans_distance'] != None: # will delete those clusters whose only has 1 instance whose distance is smaller than threshold or those that only have 1 instance
            new_code_dict[code] = {}
            retain_idx = [ii for ii, d in enumerate(code_dict[code]['kmeans_distance']) if d <= distance_threshold]
            new_code_dict[code]["kmeans_distance"] = [code_dict[code]['kmeans_distance'][iii] for iii in retain_idx]
            new_code_dict[code]["utt"] = [code_dict[code]['utt'][iii] for iii in retain_idx]
            new_code_dict[code]["boundaries"] = [code_dict[code]['boundaries'][iii] for iii in retain_idx]
            new_code_dict[code]["index"] = [code_dict[code]['index'][iii] for iii in retain_idx]
            new_code_dict[code]["vghubert_feat"] = np.stack([code_dict[code]['vghubert_feat'][iii] for iii in retain_idx], axis=0)
    code_dict = new_code_dict

################## run aggolomerative clustering on code_dict ##################
################## run aggolomerative clustering on code_dict ##################
################## run aggolomerative clustering on code_dict ##################
################## run aggolomerative clustering on code_dict ##################
################## run aggolomerative clustering on code_dict ##################
code_list = [code for code in code_dict]
if args.agglomerative_approach == "classic":
    print("....agglomerative clustering on kmeans clusters....")
    # prepare input array
    input_feat = np.stack([code_dict[code]["vghubert_feat"].mean(0) for code in code_list])
    # fit and predict
    from sklearn.cluster import AgglomerativeClustering
    assignment = AgglomerativeClustering(n_clusters=250 if args.debug else args.agglomerative_k, affinity=args.affinity, linkage=args.linkage).fit_predict(input_feat)
    # add new_code to code_dict by iterating code_list
    for i, code in enumerate(code_list):
        code_dict[code]['agg_code'] = assignment[i].item()
elif args.agglomerative_approach == "customized":
    raise NotImplementedError
    # as above, but need to have our own agg model class

# reconstruct the code_dict using agg_code as the key
new_code_dict = {}
for code in code_dict:
    agg_code = code_dict[code]['agg_code']
    if agg_code not in new_code_dict:
        new_code_dict[agg_code] = {"utt":[], "index":[], "boundaries":[], "word_boundaries":[], "locations":[], "vghubert_feat":[]}
    new_code_dict[agg_code]['utt'] += code_dict[code]['utt']
    new_code_dict[agg_code]['index'] += code_dict[code]['index']
    new_code_dict[agg_code]['boundaries'] += code_dict[code]['boundaries']
    new_code_dict[agg_code]['word_boundaries'] += [feats_dict[utt]['word_boundaries'][ind] for utt, ind in zip(code_dict[code]['utt'], code_dict[code]['index'])]
    new_code_dict[agg_code]['locations'] += [feats_dict[utt]['locations'][ind] for utt, ind in zip(code_dict[code]['utt'], code_dict[code]['index'])]
    new_code_dict[agg_code]['vghubert_feat'].append(code_dict[code]['vghubert_feat'])
for code in new_code_dict:
    new_code_dict[code]['vghubert_feat'] = np.concatenate(new_code_dict[code]['vghubert_feat'], axis=0)

code_dict = new_code_dict

data_dict = {}
for code in code_dict:
    temp = code_dict[code]
    for jj, (utt, bound, loc, word_b) in enumerate(zip(temp['utt'], temp['boundaries'], temp['locations'], temp['word_boundaries'])):
        if utt not in data_dict:
            data_dict[utt] = {"code": [], "locations": [], "boundaries": [], "word_boundaries":[], "spf": feats_dict[utt]['spf']}
        
        data_dict[utt]["code"].append(code) # it's already 'agg_code'
        data_dict[utt]['locations'].append(loc)
        data_dict[utt]['boundaries'].append(bound)
        if "refined_word_boundaries" in code_dict[code]:
            data_dict[utt]['word_boundaries'].append(code_dict[code]["refined_word_boundaries"][jj])
        else:
            data_dict[utt]['word_boundaries'].append(word_b)


print("....re-generate data_dict.pkl for unit_analysis.py....")
for utt, val in data_dict.copy().items():
    sort_ind = np.argsort(val['locations'])
    data_dict[utt]['code'] = np.array(val['code'])[sort_ind]
    data_dict[utt]['locations'] = torch.tensor(val['locations'])[sort_ind]
    data_dict[utt]['boundaries'] = torch.stack(val['boundaries'])[sort_ind]
    data_dict[utt]['word_boundaries'] = torch.tensor(val['word_boundaries'])[sort_ind].tolist()

# based on the filter, extract remaining instance from feats_dcit, save it to pkl, for unit_analysis.py
with open(os.path.join(save_root, 'data_dict.pkl'), "wb") as f:
    pickle.dump(data_dict, f)
print(f"save pickle data at {os.path.join(save_root, 'data_dict.pkl')}")