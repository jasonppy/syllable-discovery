
model=$1 # vg-hubert_3 or vg-hubert_4
tgt_layer_for_attn=$2 # 9
k=$3 # 4096
threshold=$4 # 0.7
reduce_method=$5 # mean, max etc.
segment_method=$6 # clsAttn
seed=$7 # 1
dataset=spokencoco


model_root=/path/to/root/of/model/parent_folder # i.e. the parent folder of folder vg-hubert_x
data_root=/path/to/coco/
save_root= # save intermediate data

data_json="${data_root}/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy_with_alignments.json"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate wd
python ../save_seg_feats.py \
--segment_method ${segment_method} \
--threshold ${threshold} \
--reduce_method ${reduce_method} \
--tgt_layer_for_attn ${tgt_layer_for_attn} \
--exp_dir ${model_root}/discovery/${model} \
--audio_base_path ${data_root}/SpokenCOCO \
--save_root ${save_root} \
--data_json ${data_json} \
--dataset ${dataset}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate wd
python ../run_kmeans.py \
--seed ${seed} \
--segment_method ${segment_method} \
--threshold ${threshold} \
--reduce_method ${reduce_method} \
--tgt_layer_for_attn ${tgt_layer_for_attn} \
-f "CLUS${k}" \
--exp_dir ${save_root}/${model} \
--dataset ${dataset}


source ~/miniconda3/etc/profile.d/conda.sh
conda activate wd
python ../unit_analysis.py \
--exp_dir "${save_root}/${model}/${dataset}_${reduce_method}_${threshold}_${tgt_layer_for_attn}_${segment_method}" \
--data_json ${data_json} \
--k ${k} >> "./logs/${model}_${dataset}_${reduce_method}_${threshold}_${tgt_layer_for_attn}_${segment_method}.log" 2>&1