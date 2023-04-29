import math
import argparse
import numpy as np
import torch
from torch import nn
from .audio_encoder import  AudioEncoder
from .utils import Margin_InfoNCE_loss
from .vision_transformer import vit_tiny, vit_small, vit_base
import logging
logger = logging.getLogger(__name__)

class DualEncoder(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--num_attention_heads", type=int, help="number of attention heads for visn transformer and cross-modal transformer", default=12)
        parser.add_argument("--intermediate_size", type=int, help="size of feed forward net dimension in visn transformer and cross-modal transformer", default=3072)
        parser.add_argument("--hidden_size", type=int, help="dimension of transformer feature in visn transformer and cross-modal transformer", default=768)
        parser.add_argument("--hidden_act", type=str, help="activation function of visn transformer and cross-modal transformer", default="gelu")
        parser.add_argument("--hidden_dropout_prob", type=float, help="dropout prob for visn transformer and cross-modal transformer", default=0.1)
        parser.add_argument("--attention_probs_dropout_prob", type=float, help="attention dropout prob for visn transformer and cross-modal transformer", default=0.1)
        parser.add_argument("--max_position_embeddings", type=int, default=512) # not used
        parser.add_argument("--initializer_range", type=float, help="range of linear layers (QKV layers) of visn transformer and cross-modal transformer", default=0.02)
        parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
        parser.add_argument("--xtrm_layers", type=int, help="number of cross-modal layer", default=5)
        parser.add_argument("--return_attention_weight", action="store_true", default=False, help="return the attention weight of the first layer of the first x_layer, i.e. audio attends to image feats [b,heads,T_audio,T_image]")
        parser.add_argument("--fine_matching_weight", type=float, default=1.0)
        parser.add_argument("--cls_coarse_matching_weight", type=float, default=0.1)
        parser.add_argument("--feat_coarse_matching_weight", type=float, default=0.1)
        parser.add_argument("--load_w2v2_weights", type=str, default=None)
        parser.add_argument("--load_hubert_weights", type=str, default=None)
        parser.add_argument("--load_pretrained_vit", type=str, default=None)
        parser.add_argument("--margin", type=float, default=1.0)
        parser.add_argument('--vit_arch', default='vitsmall', type=str,
        choices=['vittiny', 'vitsmall', 'vitbase'], help='Architecture (support only ViT atm).')
        parser.add_argument('--vit_patch_size', default=16, type=int, help='Patch resolution of the model.')
        parser.add_argument("--vit_checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
        parser.add_argument("--common_embed_dim", type=int, default=2048, help="embedding space size that final features/cls are projected to")
        parser.add_argument("--cls_loss", action="store_true", default=False)
        parser.add_argument("--feat_loss", action="store_true", default=False)
        parser.add_argument("--nonlinear_proj", action="store_true", default=False)
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.args.encoder_layers = self.args.layer_use + 1
        self.audio_encoder = AudioEncoder(argparse.Namespace(**vars(self.args))) # create a new instance of the arg class, in case some operation in w2v2 changes the original args
        
        if args.vit_arch == "vitsmall":
            self.trm = vit_small(patch_size=self.args.vit_patch_size, num_classes=0)
        elif args.vit_arch == "vitbase":
            self.trm = vit_base(patch_size=self.args.vit_patch_size, num_classes=0)
        else:
            raise NotImplementedError
        logger.info(f"Model {self.args.vit_arch} {self.args.vit_patch_size}x{self.args.vit_patch_size} built")
        
        if self.args.cls_loss:
            self.audio_cls_token_proj = nn.Sequential(nn.Linear(self.args.encoder_embed_dim,self.args.encoder_embed_dim*2), nn.GELU(), nn.Linear(self.args.encoder_embed_dim*2,self.args.common_embed_dim))
            self.visual_cls_token_proj = nn.Sequential(nn.Linear(self.trm.embed_dim,self.args.encoder_embed_dim*2), nn.GELU(), nn.Linear(self.args.encoder_embed_dim*2,self.args.common_embed_dim))
        if self.args.feat_loss:
            self.audio_feats_proj = nn.Sequential(nn.Linear(self.args.encoder_embed_dim,self.args.encoder_embed_dim*2), nn.GELU(), nn.Linear(self.args.encoder_embed_dim*2,self.args.common_embed_dim))
            self.visual_feats_proj = nn.Sequential(nn.Linear(self.trm.embed_dim,self.args.encoder_embed_dim*2), nn.GELU(), nn.Linear(self.args.encoder_embed_dim*2,self.args.common_embed_dim))
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    

    def forward_image(self, images):
        out = self.trm(images)
        if self.args.cls_loss:
            out['visual_cls'] = self.visual_cls_token_proj(out['visual_cls'])
        if self.args.feat_loss:
            out['visual_feats']= self.visual_feats_proj(out['visual_feats'])
        return out

    def forward_audio(self, audio_feats, audio_attention_mask, test=False):
        if test:
            self.audio_encoder.eval()
            trm2_out = self.audio_encoder(audio_feats, padding_mask=audio_attention_mask, mask=False, tgt_layer=self.args.layer_use) # pass tgt_layer if you want TransformerEncoder to stop forward at that layer, this is usually used in testing retrieval accuracy
        else:
            self.audio_encoder.train()
            trm2_out = self.audio_encoder(audio_feats, padding_mask=audio_attention_mask, mask=False, tgt_layer=self.args.layer_use) # pass tgt_layer if you want TransformerEncoder to stop forward at that layer, this is usually used in testing retrieval accuracy
        
        cls_token = trm2_out['cls_token']
        audio_feats = trm2_out['layer_feats']
        attention_mask = trm2_out['padding_mask']

        if self.args.cls_loss:
            cls_token = self.audio_cls_token_proj(cls_token)
        if self.args.feat_loss:
            audio_feats = self.audio_feats_proj(audio_feats)
        if attention_mask != None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) #[batch_size, 1, 1, to_seq_length]
            # this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            extended_audio_attention_mask = extended_attention_mask * -10000.0 # 0.0 is what we want to attend, -10000. is what we don't want to attend
        else:
            extended_audio_attention_mask = None
        return audio_feats, cls_token, extended_audio_attention_mask

    def forward(
        self,
        audio_feats,
        images,
        attention_mask=None,
        test = False,
        analysis = False
    ):
        if analysis:
            visual_out = self.forward_image(images)
            audio_feats, audio_cls, extended_audio_attention_mask = self.forward_audio(audio_feats, attention_mask, test=True)
            nframes = (extended_audio_attention_mask.squeeze(1).squeeze(1) == 0.0).to(audio_feats).sum(1) if extended_audio_attention_mask != None else None
            return audio_feats, audio_cls, nframes, visual_out['visual_feats'], visual_out['visual_cls']
        if test:
            visual_out = self.forward_image(images)
            audio_feats, audio_cls, extended_audio_attention_mask = self.forward_audio(audio_feats, attention_mask, test=True)
            nframes = (extended_audio_attention_mask.squeeze(1).squeeze(1) == 0.0).to(audio_feats).sum(1)
            return audio_feats.sum(1)/nframes.unsqueeze(-1), audio_cls, extended_audio_attention_mask, visual_out['visual_feats'].mean(1), visual_out['visual_cls']     
        else:
            visual_out = self.forward_image(images)
            audio_feats, audio_cls, extended_audio_attention_mask = self.forward_audio(audio_feats, attention_mask)
            nframes = (extended_audio_attention_mask.squeeze(1).squeeze(1) == 0.0).to(audio_feats).sum(1)
            return audio_feats.sum(1)/nframes.unsqueeze(-1), audio_cls, extended_audio_attention_mask, visual_out['visual_feats'].mean(1), visual_out['visual_cls']

    def carefully_load_state_dict(self, states):
        """
        1) Take care of DataParallel/nn.Module state_dict
        2) Show keys that are not loaded due to size mismatch or not found in model
        """
        new_states = self.state_dict()
        loaded_keys = []
        for k, v in states.items():
            k = k[7:] if k.startswith('module') else k
            # if "audio_convnet" in k:
            #     print(f"skip audio convnet weights {k}")
            #     continue
            if k in new_states and new_states[k].size() == v.size():
                new_states[k] = v
                loaded_keys.append(k)
            else:
                print('Ignoring %s due to not existing or size mismatch' % k)

        non_loaded_keys = set(new_states.keys()) - set(loaded_keys)
        if non_loaded_keys:
            print('\nDual Encoder states that do not exist in the seed_dir:')
            for k in sorted(non_loaded_keys):
                print('  %s' % k)
        
        self.load_state_dict(new_states)
