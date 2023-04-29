import time
import os
import torch
import math
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datasets import spokencoco_dataset, places_dataset
from datasets.sampler import StatefulSampler
from models import dual_encoder
from .utils import *
from .trainer_utils import *
from .bert_adam import BertAdam
from models import vit_utils
from apex.fp16_utils import *
from apex import amp
from logging import getLogger
logger = getLogger(__name__)

class Trainer:
    @staticmethod
    def add_args(parser):
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--exp_dir", type=str)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--val_batch_size", type=int)
        parser.add_argument("--val_cross_batch_size", type=int)
        parser.add_argument("--n_epochs", type=int)
        parser.add_argument("--n_print_steps", type=int)
        parser.add_argument("--n_val_steps", type=int)
        parser.add_argument("--lr", type=float, default=0.0001)
        parser.add_argument("--warmup_fraction", type=float, default=0.1)
        parser.add_argument("--opt_level", type=str, default="O0", help="O0, O1, O2, O3. O0:fp32, O1:fp16+fp32 mixed, O2:almost fp16, O3:fp16")
    
    def __init__(self, args):
        self.start_time = time.time()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"number of devices: {torch.cuda.device_count()}")
        self.writer = SummaryWriter(self.args.exp_dir)
        self.seed_everything(seed=self.args.seed)
        self.meters = self._setup_meters()
        self.progress, self.total_progress = setup_progress(self)
        self.dual_encoder, self.trainables, self.indices, self.optim_states = self._setup_models()
        self.train_loader, self.valid_loader, self.valid_loader2, self.train_sampler, self.train_data_length = self._setup_dataloader()
        self.total_num_updates = int(math.floor(self.train_data_length / self.args.batch_size))*self.args.n_epochs
        self.optimizer = self._setup_optimizer()
        self.dual_encoder, self.optimizer = amp.initialize(models=self.dual_encoder, optimizers=self.optimizer, opt_level=self.args.opt_level)
        if torch.cuda.device_count() > 1:
            self.dual_encoder = nn.DataParallel(self.dual_encoder)
        self.scheduler = self._setup_scheduler()
        self.criterion = dual_encoder.Margin_InfoNCE_loss
        logger.info(f"batch size: {self.args.batch_size}")
    
    def forward(self, batch):
        audio_feats, audio_cls, extended_audio_attention_mask, visual_feats, visual_cls, = self.dual_encoder(audio_feats = batch['audio'], attention_mask = batch['audio_attention_mask'], images = batch['images'])
        losses = {}
        if self.args.cls_loss:
            coarse_cross_relationship_score_matrix = visual_cls @ audio_cls.transpose(0,1)
            losses['cls_matching_loss'] = dual_encoder.Margin_InfoNCE_loss(coarse_cross_relationship_score_matrix, margin=self.args.margin, img_id = batch['img_id'])
        if self.args.feat_loss:
            coarse_cross_relationship_score_matrix = visual_feats @ audio_feats.transpose(0,1)
            losses['feat_matching_loss'] = dual_encoder.Margin_InfoNCE_loss(coarse_cross_relationship_score_matrix, margin=self.args.margin, img_id = batch['img_id'])
        return losses

    def train(self):
        flag = True
        step_per_epoch = int(self.train_data_length/self.args.batch_size)
        data_start_time = time.time()

        while flag:
            for i, batch in enumerate(self.train_loader):
                data_end_time = time.time()
                self.dual_encoder.train()
                if self.progress['num_updates'] > self.total_num_updates:
                    flag = False
                    self.validate_and_save()
                    self.writer.close()
                    break
                
                cur_lr = np.mean(self.optimizer.get_lr())

                self.writer.add_scalar("lr", cur_lr, self.progress['num_updates'])
                cur_step = self.progress['num_updates'] % step_per_epoch

                cur_batch = {
                        "images": batch['images'].to(self.device),
                        "audio": batch['audio'].to(self.device),
                        "audio_attention_mask": batch['audio_attention_mask'].to(self.device),
                        "img_id": batch['img_id']
                        }

                losses = self.forward(cur_batch)

                for key in losses:
                    if key in self.meters:
                        self.meters[key].update(losses[key].mean().cpu().item(), cur_batch['images'].shape[0])
                        self.writer.add_scalar(key, self.meters[key].val, self.progress['num_updates'])
                
                weighted_loss = self.weight_loss(losses)

                self.meters['weighted_loss'].update(weighted_loss.item(), cur_batch['images'].shape[0])
                self.writer.add_scalar('weighted_loss', weighted_loss.item(), self.progress['num_updates'])
                
                with amp.scale_loss(weighted_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 1.)
                # weighted_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.trainables, 1.)
                    

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.meters['data_time'].update(data_end_time - data_start_time)
                self.meters['train_time'].update(time.time() - data_end_time)
                self.writer.add_scalar("data_time", data_end_time - data_start_time, self.progress['num_updates'])
                self.writer.add_scalar("train_time", time.time() - data_end_time, self.progress['num_updates'])

                # logging
                if self.progress['num_updates'] % self.args.n_print_steps == 0:
                    log_out = {}
                    log_out['epoch'] = f"{self.progress['epoch']}/{self.args.n_epochs}"
                    log_out['cur_step/steps_per_epoch'] = f"{cur_step}/{step_per_epoch}"
                    log_out['num_updates'] = self.progress['num_updates']
                    log_out['lr'] = f"{cur_lr:.7f}"
                    for key in self.meters:
                        if self.meters[key].val != 0 or self.meters[key].avg != 0:
                            log_out[key] = f"{self.meters[key].val:.4f} ({self.meters[key].avg:.4f})" if isinstance(self.meters[key].val, float) else f"{self.meters[key].val}"
                    logger.info(log_out)
                    if np.isnan(self.meters['weighted_loss'].avg):
                        logger.info("training diverged...")
                        return
                # validation and save models
                if self.progress['num_updates'] % self.args.n_val_steps == 0:
                    self.validate_and_save(places=self.args.places)

                self.progress['num_updates'] += 1
                self.progress['epoch'] = int(math.ceil(self.progress['num_updates'] / step_per_epoch))
                data_start_time = time.time()

    def validate_and_save(self, libri=False, places=False):
        self.dual_encoder.eval()
        if places:
            r10, r5, r1 = self.validate(self.valid_loader)
            r10_unseen, r5_unseen, r1_unseen = self.validate(self.valid_loader2, unseen=True)
            r10, r5, r1 = (r10+r10_unseen)/2, (r5+r5_unseen)/2, (r1+r1_unseen)/2
        else:
            r10, r5, r1 = self.validate(self.valid_loader)
        
        # r1 = 0.1 # ignore validation, for debugging
        if r1 > self.progress['best_acc']:
            self.progress['best_epoch'] = self.progress['epoch']
            self.progress['best_acc'] = r1
            save_path = os.path.join(self.args.exp_dir,"best_bundle.pth")
            torch.save(
                {
                    "dual_encoder": self.dual_encoder.module.state_dict() if torch.cuda.device_count() > 1 else self.dual_encoder.state_dict(),
                    "optimizer":  self.optimizer.state_dict(),
                    "indices": self.train_sampler.state_dict()
                },save_path
            )
            logger.info(f"save *best* models at {save_path} at global step {self.progress['num_updates']}")
        save_progress(self)
        save_path = os.path.join(self.args.exp_dir,"bundle.pth")
        torch.save(
            {
                "dual_encoder": self.dual_encoder.module.state_dict() if torch.cuda.device_count() > 1 else self.dual_encoder.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "indices": self.train_sampler.state_dict()
            },save_path
        )
        logger.info(f"save models, indices, acc and other statistics at {save_path} and {self.args.exp_dir}/progress.pkl at global step {self.progress['num_updates']}")

    def validate(self, valid_loader=None, unseen=False, hide_progress=True):
        if valid_loader == None:
            valid_loader = self.valid_loader
        self.dual_encoder.eval()

        start_val_time = time.time()
        N_examples = valid_loader.dataset.__len__()

        # frame_counts = []
        with torch.no_grad():
            # get single modal representations
            audio_feats_total = [] 
            extended_audio_attention_mask_total = []
            audio_cls_total = []
            audio_img_id_total = [] # this is same order as audio_cls_total and audio_feats_total
            img_id_to_img_feats = {}
            img_img_id_list = []
            img_cls_list = [] # this is distinct, order is the same as img_img_id_list
            img_feats_list = [] # this is distinct, order is the same as img_img_id_list
            for i, batch in enumerate(valid_loader):
                self.dual_encoder.eval()
                
                audio_feats, audio_cls, extended_audio_attention_mask, visual_feats, visual_cls = self.dual_encoder(audio_feats = batch['audio'].to(self.device), attention_mask = batch['audio_attention_mask'].to(self.device), images = batch['images'].to(self.device), test = True)
                audio_cls_total.append(audio_cls)
                audio_feats_total.append(audio_feats)
                
                extended_audio_attention_mask_total.append(extended_audio_attention_mask)
                audio_img_id_total.append(batch['img_id'])
                for i, img_id in enumerate(batch['img_id']):
                    if img_id not in img_id_to_img_feats:
                        img_id_to_img_feats[img_id] = 1
                        img_feats_list.append(visual_feats[i])
                        img_cls_list.append(visual_cls[i] if visual_cls is not None else None)
                        img_img_id_list.append(img_id)
                
            logger.info(f"time can be cached: {time.time() - start_val_time:.3f}")
            audio_cls_total = torch.cat(audio_cls_total) if audio_cls_total[0] is not None else None
            img_cls_list = torch.stack(img_cls_list) if img_cls_list[0] is not None else None
            audio_feats_total = torch.cat(audio_feats_total)
            img_feats_list = torch.stack(img_feats_list)
            extended_audio_attention_mask_total = torch.cat(extended_audio_attention_mask_total)
            audio_img_id_total = np.concatenate(audio_img_id_total)
            img_img_id_list = np.array(img_img_id_list)
            if self.args.cls_loss:
                coarse_cross_relationship_score_matrix = img_cls_list @ audio_cls_total.transpose(0,1)
                recalls_cls = calc_recalls_from_S_one_to_many_coarse(coarse_cross_relationship_score_matrix, row_img_id=img_img_id_list, column_img_id=audio_img_id_total)
                avg_acc_coarse = (recalls_cls['A_r10'] + recalls_cls['I_r10']) / 2
                avg_acc_r1_coarse = (recalls_cls['A_r1'] + recalls_cls['I_r1']) / 2
                if unseen:
                    logger.info("UNSEEN UNSEEN UNSEEN")
                    self.writer.add_scalar("acc_coarse_cls_unseen", avg_acc_coarse, self.progress['num_updates'])
                    self.writer.add_scalar("acc_r1_coarse_cls_unseen", avg_acc_r1_coarse, self.progress['num_updates'])
                else:
                    self.writer.add_scalar("acc_coarse_cls", avg_acc_coarse, self.progress['num_updates'])
                    self.writer.add_scalar("acc_r1_coarse_cls", avg_acc_r1_coarse, self.progress['num_updates'])
                logger.info("Using [ClS] for Coarse Retrieval Accuracy:")
                logger.info('Audio R@100 {A_r100:.3f} Image R@100 {I_r100:.3f} Average R@100 {r100_ave:.3f} over {N:d} validation pairs'.format(A_r100=recalls_cls['A_r100'], I_r100=recalls_cls['I_r100'], r100_ave=(recalls_cls['A_r100']+recalls_cls['I_r100'])/2, N=N_examples))
                logger.info('Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} Average R@10 {r10_ave:.3f} over {N:d} validation pairs'.format(A_r10=recalls_cls['A_r10'], I_r10=recalls_cls['I_r10'], r10_ave=(recalls_cls['A_r10']+recalls_cls['I_r10'])/2, N=N_examples))
                logger.info('Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} Average R@5 {r5_ave:.3f} over {N:d} validation pairs'.format(A_r5=recalls_cls['A_r5'], I_r5=recalls_cls['I_r5'], r5_ave=(recalls_cls['A_r5']+recalls_cls['I_r5'])/2, N=N_examples))
                logger.info('Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} Average R@1 {ave_r1:.3f} over {N:d} validation pairs'.format(A_r1=recalls_cls['A_r1'], I_r1=recalls_cls['I_r1'], ave_r1=(recalls_cls['A_r1']+recalls_cls['I_r1'])/2,  N=N_examples))
                logger.info(f"validation time: {time.time() - start_val_time:.3f}")
            else:
                recalls_cls = None
            if self.args.feat_loss:
                coarse_cross_relationship_score_matrix = img_feats_list @ audio_feats_total.transpose(0,1)
                recalls_feat = calc_recalls_from_S_one_to_many_coarse(coarse_cross_relationship_score_matrix, row_img_id=img_img_id_list, column_img_id=audio_img_id_total)
                avg_acc_coarse = (recalls_feat['A_r10'] + recalls_feat['I_r10']) / 2
                avg_acc_r1_coarse = (recalls_feat['A_r1'] + recalls_feat['I_r1']) / 2
                if unseen:
                    logger.info("UNSEEN UNSEEN UNSEEN")
                    self.writer.add_scalar("acc_coarse_feat_unseen", avg_acc_coarse, self.progress['num_updates'])
                    self.writer.add_scalar("acc_r1_coarse_feat_unseen", avg_acc_r1_coarse, self.progress['num_updates'])
                else:
                    self.writer.add_scalar("acc_coarse_feat", avg_acc_coarse, self.progress['num_updates'])
                    self.writer.add_scalar("acc_r1_coarse_feat", avg_acc_r1_coarse, self.progress['num_updates'])
                logger.info("Using mean-pooled feat for Coarse Retrieval Accuracy:")
                logger.info('Audio R@100 {A_r100:.3f} Image R@100 {I_r100:.3f} Average R@100 {r100_ave:.3f} over {N:d} validation pairs'.format(A_r100=recalls_feat['A_r100'], I_r100=recalls_feat['I_r100'], r100_ave=(recalls_feat['A_r100']+recalls_feat['I_r100'])/2, N=N_examples))
                logger.info('Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} Average R@10 {r10_ave:.3f} over {N:d} validation pairs'.format(A_r10=recalls_feat['A_r10'], I_r10=recalls_feat['I_r10'], r10_ave=(recalls_feat['A_r10']+recalls_feat['I_r10'])/2, N=N_examples))
                logger.info('Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} Average R@5 {r5_ave:.3f} over {N:d} validation pairs'.format(A_r5=recalls_feat['A_r5'], I_r5=recalls_feat['I_r5'], r5_ave=(recalls_feat['A_r5']+recalls_feat['I_r5'])/2, N=N_examples))
                logger.info('Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} Average R@1 {ave_r1:.3f} over {N:d} validation pairs'.format(A_r1=recalls_feat['A_r1'], I_r1=recalls_feat['I_r1'], ave_r1=(recalls_feat['A_r1']+recalls_feat['I_r1'])/2,  N=N_examples))
                logger.info(f"validation time: {time.time() - start_val_time:.3f}")
            else:
                recalls_feat = None
        count = 0
        avg_acc_r10 = 0.
        avg_acc_r5 = 0.
        avg_acc_r1 = 0.
        if recalls_cls is not None:
            avg_acc_r10 += (recalls_cls['A_r10'] + recalls_cls['I_r10'])
            avg_acc_r5 += (recalls_cls['A_r5'] + recalls_cls['I_r5'])
            avg_acc_r1 += (recalls_cls['A_r1'] + recalls_cls['I_r1'])
            count += 2

        if recalls_feat is not None and recalls_cls is None: # since cls retrieval is always better, if using cls feat, calculate retrieval based only on it.
            avg_acc_r10 += (recalls_feat['A_r10'] + recalls_feat['I_r10'])
            avg_acc_r5 += (recalls_feat['A_r5'] + recalls_feat['I_r5'])
            avg_acc_r1 += (recalls_feat['A_r1'] + recalls_feat['I_r1'])
            count += 2

        avg_acc_r10 = avg_acc_r10 / count
        avg_acc_r5 = avg_acc_r5 / count
        avg_acc_r1 = avg_acc_r1 / count
        if unseen:
            self.writer.add_scalar("acc_r10_unseen", avg_acc_r10, self.progress['num_updates'])
            self.writer.add_scalar("acc_r5_unseen", avg_acc_r5, self.progress['num_updates'])
            self.writer.add_scalar("acc_r1_unseen", avg_acc_r1, self.progress['num_updates'])
        else:
            self.writer.add_scalar("acc_r10", avg_acc_r10, self.progress['num_updates'])
            self.writer.add_scalar("acc_r5", avg_acc_r5, self.progress['num_updates'])
            self.writer.add_scalar("acc_r1", avg_acc_r1, self.progress['num_updates'])
        return avg_acc_r10, avg_acc_r5, avg_acc_r1

    def _setup_meters(self):
        meters = {}
        meter_names = ['weighted_loss',  "cls_matching_loss", "feat_matching_loss",'data_time', 'train_time']
        for name in meter_names:
            meters[name] = AverageMeter()
        return meters
    
    def _setup_models(self):
        model = dual_encoder.DualEncoder(self.args)
        logger.info(model)
        print_model_info(model)
        if self.args.validate:
            bundle = torch.load(os.path.join(self.args.exp_dir, "best_bundle.pth"))
            model.carefully_load_state_dict(bundle['dual_encoder'])
            indices = None
            optim_states = None
            # logger.info("loaded parameters and data indices from epoch %d, global step %d" % (self.progress['epoch'], self.progress['num_updates']))
            logger.info("Perform Validation")
        elif self.progress['num_updates'] > 1:
            bundle = torch.load(os.path.join(self.args.exp_dir, "bundle.pth"))
            model.carefully_load_state_dict(bundle['dual_encoder'])
            indices = bundle['indices']
            optim_states = bundle['optimizer']
            logger.info("loaded parameters and data indices from epoch %d, global step %d" % (self.progress['epoch'], self.progress['num_updates']))
        else:
            indices = None
            optim_states = None

        if self.args.load_w2v2_weights != None and self.progress['num_updates'] <= 1 and not self.args.validate:
            model.audio_encoder.carefully_load_state_dict(torch.load(self.args.load_w2v2_weights)['model'])
        elif self.args.load_hubert_weights != None and self.progress['num_updates'] <= 1 and not self.args.validate:
            model.audio_encoder.carefully_load_state_dict(torch.load(self.args.load_hubert_weights)['model'])

        if self.args.feature_grad_mult <= 0.:
            for name, p in model.named_parameters():
                if "feature_extractor" in name:
                    p.requires_grad = False
        if self.args.freeze_first_x != None:
            freeze_names =  [f'audio_encoder.encoder.layers.{i}.' for i in range(self.args.freeze_first_x)]
            for n, p in model.named_parameters():
                for fn in freeze_names:
                    if n.startswith(fn):
                        if p.requires_grad:
                            p.requires_grad = False
                            print(f"disable gradient of weights: {n}")
                            break

        if self.args.load_pretrained_vit != None and self.progress['num_updates'] <= 1 and not self.args.validate:
            ckpt_root = self.args.load_pretrained_vit
            ckpt_name = f"dino_{self.args.vit_arch.lower()}{str(self.args.vit_patch_size)}_pretrain_full_checkpoint.pth"
            ckpt_fn = os.path.join(ckpt_root, ckpt_name)
            vit_utils.load_pretrained_weights(model.trm, ckpt_fn, self.args.vit_checkpoint_key, self.args.vit_arch, self.args.vit_patch_size)

        trainables = [p for p in model.parameters() if p.requires_grad]

        model.to(self.device)

        return model, trainables, indices, optim_states

    def _setup_dataloader(self):
        if self.args.places:
            # raise NotImplementedError
            train_dataset = places_dataset.ImageCaptionDataset(self.args, split='train')
            val_seen_dataset = places_dataset.ImageCaptionDataset(self.args, split='val_seen')
            val_unseen_dataset = places_dataset.ImageCaptionDataset(self.args, split='val_unseen')
            train_sampler = StatefulSampler(len(train_dataset))
            if self.progress['num_updates'] > 1 and self.indices is not None:
                train_sampler.load_state_dict(self.indices)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True, sampler = train_sampler, collate_fn = train_dataset.collate, drop_last=True)
            valid_loader = torch.utils.data.DataLoader(val_seen_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, collate_fn = val_seen_dataset.collate)
            valid_loader2 = torch.utils.data.DataLoader(val_unseen_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, collate_fn = val_unseen_dataset.collate)
        else:
        # SpokenCOCO
            train_dataset = spokencoco_dataset.ImageCaptionDataset(self.args, split='train')
            val_dataset = spokencoco_dataset.ImageCaptionDataset(self.args, split='val')
            train_sampler = StatefulSampler(len(train_dataset))
            if self.progress['num_updates'] > 1 and self.indices is not None:
                train_sampler.load_state_dict(self.indices)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True, sampler = train_sampler, collate_fn = train_dataset.collate, drop_last=True)
            valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, collate_fn = val_dataset.collate)
            valid_loader2 = None

        return train_loader, valid_loader, valid_loader2, train_sampler, len(train_dataset)
    
    def _setup_optimizer(self):
        optimizer = BertAdam(self.trainables, lr=self.args.lr, warmup=self.args.warmup_fraction, t_total=self.total_num_updates)

        if self.progress['num_updates'] > 1:
            optimizer.load_state_dict(self.optim_states)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        optimizer.zero_grad()
        return optimizer
    
    def _setup_scheduler(self):
        pass

    def weight_loss(self, losses):
        weighted_loss = 0.
        if "cls_matching_loss" in losses:
            weighted_loss += losses['cls_matching_loss'] * self.args.cls_coarse_matching_weight
        if "feat_matching_loss" in losses:
            weighted_loss += losses['feat_matching_loss'] * self.args.feat_coarse_matching_weight
        
        return weighted_loss
    
    def seed_everything(self, seed=1):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

