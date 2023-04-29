
model=$1 # vg-hubert_3
tgt_layer_for_attn=$2 # 9
k=$3 # 16384
threshold=$4 # 0.7
reduce_method=$5 # max
segment_method=$6 # clsAttn
seed=$7 # 1
snapshot=$8 # 20
language=$9 # english, mandarin, french, lang1, lang2
insert_threshold=${12} # 10000 if no insertion
dataset=zs20

# model_root=/path/to/root/of/model/parent_folder # i.e. the parent folder of folder vg-hubert_x
# data_root=/zs20/
# save_root=/path/to/save_data

save_root=/data2/scratch/pyp/exp_pyp
model_root=/data1/scratch/exp_pyp/discovery # i.e. the parent folder of folder vg-hubert_x
data_root=/data2/scratch/datasets

# running example:
# bash zs20.sh vg-hubert_3 9 16384 max clsAttn 1 20 mandarin 10000

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf2
python ../save_seg_feats_zs20_vadcon.py \
--data_root ${data_root} \
--segment_method ${segment_method} \
--threshold ${threshold} \
--reduce_method ${reduce_method} \
--tgt_layer_for_attn ${tgt_layer_for_attn} \
--exp_dir ${model_root}/${model} \
--dataset ${dataset} \
--language ${language} \
--snapshot ${snapshot} \
--insert_threshold ${insert_threshold}


source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf2
python ../kmeans_zs20_vadcon.py \
--seed ${seed} \
--segment_method ${segment_method} \
--threshold ${threshold} \
--reduce_method ${reduce_method} \
--tgt_layer_for_attn ${tgt_layer_for_attn} \
-f "CLUS${k}" \
--exp_dir ${save_root}/${model} \
--dataset ${dataset} \
--language ${language} \
--snapshot ${snapshot} \
--insert_threshold ${insert_threshold}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf2
python ../prepare_zs20_submit.py \
--exp_dir "${save_root}/${model}/${dataset}_${feats_type}_${reduce_method}_${threshold}_${tgt_layer_for_attn}_${segment_method}_${language}_snapshot${snapshot}_insertThreshold${insert_threshold}" \
--save_root ${save_root} \
--k ${k} \
--language ${language} \
--snapshot ${snapshot} \
--insert_threshold ${insert_threshold}


echo "run zerospeech 2020 evaluation"
cd ~/zerospeech2020
source ~/miniconda3/etc/profile.d/conda.sh
conda activate zerospeech2020
export ZEROSPEECH2020_DATASET=${data_root}/2020/2017
zerospeech2020-evaluate 2017-track2 ${save_root}/zs2020_snapshot${snapshot}_insertThreshold${insert_threshold} -l ${language} -o "${language}_${snapshot}_${model}_${tgt_layer_for_attn}_${segment_method}_${insert_threshold}_${feats_type}_${reduce_method}_${threshold}_${k}.json"