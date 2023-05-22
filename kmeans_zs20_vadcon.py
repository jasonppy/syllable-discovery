import argparse
import os
import os.path as osp
import time
import numpy as np
import faiss
import pickle
from collections import namedtuple
import tqdm
import json
    
print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, default='zs20')
parser.add_argument("--language", type=str,  default='english', choices=['english', 'french', 'LANG1', 'LANG2', 'mandarin'])
parser.add_argument("--exp_dir", type=str, default="/data2/scratch/pyp/discovery/word_unit_discovery/disc-16",help="directory to dump experiments")
parser.add_argument("--batch_size", type=int, default=40000)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--max_iter", type=int, default=100)
parser.add_argument("--threshold", type=float, default=0.90)
parser.add_argument("--reduce_method", type=str, default="mean", choices=['mean', 'max', 'median', 'weightedmean'])
parser.add_argument("--tgt_layer_for_attn", type=int, default=7, help="where attn weights are coming from, as for features, if feats_type==preFeats, and feature comes from previous layer of tgt_layer_for_attn, otherwise, feature comes from the same layer")
parser.add_argument("--segment_method", type=str, choices=['clsAttn', 'forceAlign'], default=None, help="if use cls attn segmentation or use force alignment segmentation. If use, need model_args.use_audio_cls_token to be True")
parser.add_argument('--faiss-specs', '-f', type=str,
                        help='faiss index specs; separated by space '
                             'format is: PCAx_NORM_CLUSx_SPHERICAL -> '
                                'PCAx if exists first apply PCA '
                                'NORM if exists, normalize the vector by L2 norm '
                                'CLUSx must exist, cluster to x clusters '
                                'SPEHRICAL if exists, apply spherical kmeans',
                        default='l2')
parser.add_argument("--seed", type=int, default=1, help="random seed for clustering")
parser.add_argument("--snapshot", type=str, default='best', help='which model snapshot to use, best means best_boundle.pth, can also pass number x, say 24, then will take snapshot_24.pth')
parser.add_argument("--insert_threshold", type=float, default=10000.0, help="if the gap between two attention segments are above the threshold, we insert a two frame segment in the middle")

args = parser.parse_args()

feats_type = args.dataset + "_" + args.reduce_method + "_" + str(args.threshold) + "_" + str(args.tgt_layer_for_attn) + "_" + args.segment_method + "_" + args.language + "_" + "snapshot"+args.snapshot + "_" + "insertThreshold" + str(args.insert_threshold if args.insert_threshold < 100 else int(args.insert_threshold))

exp_dir = osp.join(args.exp_dir, feats_type)
if not os.path.isdir(exp_dir):
    raise RuntimeError(f"{exp_dir} does not exist!!")
km_exp_dir = osp.join(exp_dir, 'kmeans_models')
os.makedirs(km_exp_dir, exist_ok=True)



faiss_spec = namedtuple("faiss_spec", ["pca", "norm", "n_clus", "sphere", "spec_str"])


def parse_faiss_specs(specs_str):
    specs = []
    for ss in specs_str.split():
        comps = ss.split("_")
        pca = 0
        norm = False
        n_clus = 0
        sphere = False
        for c in comps:
            if c.startswith("PCA"):
                pca = int(c[3:])
            elif c == "NORM":
                norm = True
            elif c.startswith("CLUS"):
                n_clus = int(c[4:])
            elif c == "SPHERICAL":
                sphere = True
        assert n_clus > 0
        specs.append(
            faiss_spec(pca=pca, norm=norm, n_clus=n_clus, sphere=sphere, spec_str=ss)
        )
    return specs

faiss_specs = parse_faiss_specs(args.faiss_specs)
print("Faiss Specs:", faiss_specs)
for spec in faiss_specs: # this is a little strange, but I guess 
    print("Processing spec", spec)

    start_time = time.time()
    with open(osp.join(exp_dir, "data_dict.pkl"), "rb") as f:
        feats_dict = pickle.load(f)
    feats = []
    for key in feats_dict:
        feats.append(feats_dict[key]['seg_feats'].numpy())
    feats = np.concatenate(feats)
    print("feature reading time: ", time.time() - start_time)
    print("FAISS KMeans training data shape: ", feats.shape)
    save_path = osp.join(km_exp_dir, spec.spec_str)
    os.makedirs(save_path, exist_ok=True)
    d = feats.shape[-1]
    x = feats
    if spec.pca > 0:
        raise NotImplementedError
        print("Computing PCA")
        pca = faiss.PCAMatrix(d, spec.pca)
        pca.train(x)
        d = spec.pca
        b = faiss.vector_to_array(pca.b)
        A = faiss.vector_to_array(pca.A).reshape(pca.d_out, pca.d_in)
        np.save(osp.join(save_path, "pca_A"), A.T)
        np.save(osp.join(save_path, "pca_b"), b)
        print("Applying PCA")
        x = pca.apply_py(x)

    if spec.norm:
        reload = spec.pca <= 0
        print("Normalizing")
        faiss.normalize_L2(x)

    print("Computing kmeans")
    kmeans = faiss.Kmeans(
        d,
        spec.n_clus,
        niter=100,
        verbose=True,
        spherical=spec.sphere,
        max_points_per_centroid=feats.shape[0],
        gpu=True,
        nredo=5,
        seed = args.seed
    )
    kmeans.train(x)
    np.save(osp.join(save_path, "centroids"), kmeans.centroids)

# assign codes
print("....assign kmeans clusters....")
code_dict = {}
for j, key in enumerate(tqdm.tqdm(feats_dict, disable=True)):
    cluster_distances, cluster_indices = kmeans.assign(feats_dict[key]['seg_feats'].numpy())
    code_dict[key] = {"boundaries": feats_dict[key]['boundaries'].tolist(), "codes": cluster_indices.tolist()}
with open(osp.join(exp_dir, "code_dict.json"), "w") as f:
    json.dump(code_dict, f)
    
