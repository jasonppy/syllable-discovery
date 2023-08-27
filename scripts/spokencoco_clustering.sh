model=$1 # vg-hubert_3
secPerSyllable=$2 # 0.2
layer=$3 # 8
kmeans_k=$4 # 16384, this is for kmeans
reduce_method=$5 # mean
segment_method=$6 # minCut, minCutMerge-0.3
seed=$7 # 1
snapshot=${8} # either 'best', or some number between 0 and 85 (inclusive)
n_proc=${9}

distance_quantile=${10} # 1.1, do not drop
cluster_drop=${11} # 0, do not drop
agglomerative_approach=${12} # classic
agglomerative_k=${13} # 4096
affinity=${14} # 'cosine'
linkage=${15} # 'average'
shift=${16} # -0.02, this is selected by running spokencoco_boundary.sh

dataset=spokencoco # (lowercase) spokencoco, timit

# running example
# sps=0.2
# layer=8
# mergeThres=0.3
# bash spokencoco_clustering.sh vg-hubert_3 ${sps} ${layer} 16384 mean minCutMerge-${mergeThres} 1 best 32 1.1 0 classic 4096 cosine average -0.02

model_root=/path/to/root/of/model/parent_folder # i.e. the parent folder of folder vg-hubert_x
data_root=/path/to/coco/
save_root= # save intermediate data



data_json="${data_root}/SpokenCOCO/SpokenCOCO_test_unrolled_karpathy_with_alignments_new_adapt_syllable_multisyllabic.json"

save_root="${save_root}/syllable_discovery/exps/${model}"
whole_feats_type="${dataset}_${segment_method}_secPerSyllable${secPerSyllable}_${layer}_${reduce_method}_snapshot${snapshot}"



# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate new_sd
# python ../save_seg_feats_mincut.py \
# --secPerSyllable ${secPerSyllable} \
# --n_proc ${n_proc} \
# --snapshot ${snapshot} \
# --segment_method ${segment_method} \
# --reduce_method ${reduce_method} \
# --layer ${layer} \
# --exp_dir ${model_root}/${model} \
# --audio_base_path ${data_root}/SpokenCOCO \
# --save_root ${save_root} \
# --data_json ${data_json}

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate new_sd
# python ../kmeans_agglomerative.py \
# --data_dict_root ${save_root}/${whole_feats_type} \
# --wav_root "${data_root}/SpokenCOCO" \
# --seed ${seed} \
# --kmeans_k ${kmeans_k} \
# --data_dict_fn "data_dict.pkl" \
# --distance_quantile ${distance_quantile} \
# --cluster_drop ${cluster_drop} \
# --agglomerative_approach ${agglomerative_approach} \
# --agglomerative_k ${agglomerative_k} \
# --affinity ${affinity} \
# --linkage ${linkage}



source ~/miniconda3/etc/profile.d/conda.sh
conda activate new_sd
python ../unit_analysis_mwm.py \
--shift ${shift} \
--tolerance 0.05 \
--exp_dir "${save_root}/${whole_feats_type}/kmeansK${kmeans_k}_distanceQuantile${distance_quantile}_clusterDrop${cluster_drop}_aggApproach${agglomerative_approach}_aggK${agglomerative_k}_affinity${affinity}_linkage${linkage}" \
--data_json ${data_json} \
--k ${agglomerative_k} >> "./logs/clustering_multisyllabic_${whole_feats_type}_kmeansK${kmeans_k}_distanceQuantile${distance_quantile}_clusterDrop${cluster_drop}_aggApproach${agglomerative_approach}_aggK${agglomerative_k}_affinity${affinity}_linkage${linkage}.log" 2>&1
