model=$1 # vg-hubert_3
secPerSyllable=$2 # 0.2
layer=$3 # 8
reduce_method=$4 # mean, max, or weightedmean
segment_method=$5 # minCutMerge-0.3
seed=$6 # 1
snapshot=$7 # either 'best', or some number between 0 and 85 (inclusive)
n_proc=$8
level=syllable
tol=0.05
dataset=spokencoco

# example command:
# sps=0.2
# layer=8
# mergeThres=0.3
# bash spokencoco_boundary.sh vg-hubert_3 ${sps} ${layer} mean minCutMerge-${mergeThres} 1 best 32


model_root=/path/to/root/of/model/parent_folder # i.e. the parent folder of folder vg-hubert_x
data_root=/path/to/coco/
save_root= # save intermediate data



data_json="${data_root}/SpokenCOCO/SpokenCOCO_test_unrolled_karpathy_with_alignments_new_adapt_syllable.json"

save_root="${save_root}/syllable_discovery/exps/${model}"
whole_feats_type="${dataset}_${segment_method}_secPerSyllable${secPerSyllable}_${layer}_${reduce_method}_snapshot${snapshot}"


source ~/miniconda3/etc/profile.d/conda.sh
conda activate new_sd
python ../save_seg_feats_mincut.py \
--secPerSyllable ${secPerSyllable} \
--n_proc ${n_proc} \
--snapshot ${snapshot} \
--segment_method ${segment_method} \
--reduce_method ${reduce_method} \
--layer ${layer} \
--exp_dir ${model_root}/${model} \
--audio_base_path ${data_root}/SpokenCOCO \
--save_root ${save_root} \
--data_json ${data_json}



source ~/miniconda3/etc/profile.d/conda.sh
conda activate new_sd
python ../unit_analysis_boundary_only.py \
--level ${level} \
--tolerance ${tol} \
--exp_dir "${save_root}/${whole_feats_type}" \
--data_json ${data_json} >> "./logs/boundary_${whole_feats_type}.log" 2>&1