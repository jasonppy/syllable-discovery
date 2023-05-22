
model=$1 # vg-hubert_3
secPerSyllable=$2 # 0.15
layer=$3 # 8
reduce_method=$4 # mean, max, or weightedmean
segment_method=$5 # minCutMerge-0.4
seed=$6 # 1
snapshot=$7 # either 'best', or some number between 0 and 85 (inclusive)
n_proc=$8
dataset=$9 # estonian or estonainval, the former is the testset
level=syllable
tol=0.05

# example command:
# sps=0.15
# layer=8
# mergeThres=0.4
# bash estonian_boundary.sh vg-hubert_3 ${sps} ${layer} mean minCutMerge-${mergeThres} 1 best 32 estonian

model_root=/path/to/model # i.e. the parent folder of folder vg-hubert_x
save_root=/path/to/intermediatesave # save intermediate data
valid_pkl_fn=/path/to/valid_pkl # include the pkl filename
valid_wav_path=/path/to/data
test_root=/path/to/test/data



save_root="${save_root}/syllable_discovery/exps/${model}"
whole_feats_type="${dataset}_${segment_method}_secPerSyllable${secPerSyllable}_${layer}_${reduce_method}_snapshot${snapshot}"

mkdir -p ./logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sd
python ../estonian.py \
--valid_pkl_fn ${valid_pkl_fn} \
--valid_wav_path ${valid_wav_path} \
--test_root ${test_root} \
--secPerSyllable ${secPerSyllable} \
--n_proc ${n_proc} \
--snapshot ${snapshot} \
--segment_method ${segment_method} \
--reduce_method ${reduce_method} \
--layer ${layer} \
--exp_dir ${model_root}/${model} \
--save_root ${save_root} \
--dataset ${dataset} >> ./logs/${whole_feats_type}.log 2>&1
