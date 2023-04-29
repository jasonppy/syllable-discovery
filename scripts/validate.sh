#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wd
export CUDA_VISIBLE_DEVICES=0,1

pretrained_root=/path/to/pretrained/hubertAndDINO
model_root=/path/to/root/of/model/parent_folder # i.e. the parent folder of folder vg-hubert_x
data_root=/path/to/coco/

python \
../run_spokencoco.py \
--random_init_last_x 3 \
--load_hubert_weights ${pretrained_root}/hubert_base_ls960.pt \
--load_pretrained_vit ${pretrained_root} \
--nonlinear_proj \
--opt_level O1 \
--cls_loss \
--use_audio_cls_token \
--audio_feat_len 8 \
--num_workers 4 \
--train_audio_dataset_json_file ${data_root}/SpokenCOCO/SpokenCOCO_train_unrolled_karpathy.json \
--val_audio_dataset_json_file ${data_root}/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy.json \
--exp_dir ${model_root}/vg-hubert_3 \
--batch_size 100 \
--val_batch_size 100 \
--n_epochs 30 \
--n_print_steps 2000 \
--n_val_steps 4000 \
--lr 0.00005 \
--warmup_fraction 0.1 \
--vit_arch 'vitsmall' \
--vit_patch_size 8 \
--vit_checkpoint_key 'teacher' \
--cls_coarse_matching_weight 1.0 \
--feature_grad_mult 0. \
--encoder_layers 12 \
--layer_use 11