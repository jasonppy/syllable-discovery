
model=$1 # vg-hubert_3 or vg-hubert_4
tgt_layer_for_attn=$2 # 9
threshold=$3 # 0.7 or 0.8
reduce_method=$4 # mean, this won't affect results
segment_method=$5 # clsAttn
dataset=$6 # (lowercased) buckeyeval or buckeyetest

model_root=/path/to/root/of/model/parent_folder # i.e. the parent folder of folder vg-hubert_x
data_root=/Buckeye

source ~/miniconda3/etc/profile.d/conda.sh
conda activate wd
python ../save_seg_feats_buckeye.py \
--data_root ${data_root} \
--segment_method ${segment_method} \
--threshold ${threshold} \
--reduce_method ${reduce_method} \
--tgt_layer_for_attn ${tgt_layer_for_attn} \
--exp_dir ${model_root}/${model} \
--dataset ${dataset}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate wd
python ../buckeye_eval.py \
--data_root ${data_root} \
--segment_method ${segment_method} \
--threshold ${threshold} \
--reduce_method ${reduce_method} \
--tgt_layer_for_attn ${tgt_layer_for_attn} \
--dataset ${dataset}

