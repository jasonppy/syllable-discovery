
model=$1 #vg-hubert_3 or vg-hubert_4
tgt_layer_for_attn=$2 # 9
k=$3 # 4096
threshold=$4 # 0.7
reduce_method=$5 # weightedmean
segment_method=$6 # clsAttn
seed=$7 # 1
dataset=zs21

model_root=/path/to/root/of/model/parent_folder # i.e. the parent folder of folder vg-hubert_x
data_root=/zs20/



source ~/miniconda3/etc/profile.d/conda.sh
conda activate wd
python ../save_seg_feats_zs21.py \
--data_root ${data_root} \
--segment_method ${segment_method} \
--threshold ${threshold} \
--reduce_method ${reduce_method} \
--tgt_layer_for_attn ${tgt_layer_for_attn} \
--exp_dir ${model_root}/${model} \
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
--exp_dir ${data_root}/${model} \
--dataset ${dataset}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate wd
python ../prepare_zs21_submit.py \
--exp_dir "${data_root}/${model}/${dataset}_${reduce_method}_${threshold}_${tgt_layer_for_attn}_${segment_method}" \
--k ${k} \
--out_dir ${data_root}/2017/track2 \

echo "run zerospeech 2020 evaluation"
cd ~/zerospeech2020
source ~/miniconda3/etc/profile.d/conda.sh
conda activate zerospeech2020
export ZEROSPEECH2020_DATASET=${data_root}/2020/2017
zerospeech2020-evaluate 2017-track2 ${data_root} -l english -o "english_${model}_${reduce_method}_${threshold}_${tgt_layer_for_attn}_${segment_method}_${k}.json"