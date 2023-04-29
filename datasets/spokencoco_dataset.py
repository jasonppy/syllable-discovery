import json
import random
import numpy as np
import os
import torch
import torch.nn.functional
import random
import soundfile as sf
from torch.utils.data import Dataset
import pickle
import torchvision.transforms as transforms
from PIL import Image
import logging
logger = logging.getLogger(__name__)

class ImageCaptionDataset(Dataset):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--train_audio_dataset_json_file", type=str, default="/data1/scratch/coco_pyp/SpokenCOCO/SpokenCOCO_train_unrolled_karpathy.json")
        parser.add_argument("--val_audio_dataset_json_file", type=str, default="/data1/scratch/coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy.json")
        parser.add_argument("--audio_feat_len", type=float, help="maximal audio length", default=8)
        parser.add_argument("--val_audio_feat_len", type=float, help="maximal audio length", default=10.)
        parser.add_argument("--normalize", action="store_true", default=False, help="whether or not normalize raw input, both w2v2 and hubert base doesn't normalize the input, but in exps in two papers, we normalized it, hopefully this doesn't make a difference")

    def __init__(self, args, split = "train"):
        self.args = args
        self.split = split
        self.audio_feat_len = args.audio_feat_len if "train" in split else args.val_audio_feat_len
        if split == "train":
            audio_dataset_json_file = args.train_audio_dataset_json_file   
        elif split == "val":
            audio_dataset_json_file = args.val_audio_dataset_json_file

        self.audio_base_path = os.path.dirname(audio_dataset_json_file)
        self.image_base_path = "/".join(audio_dataset_json_file.split("/")[:-2])

        with open(audio_dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        
        if "train" not in split:
            self.image_transform = transforms.Compose(
                [transforms.Resize(256, interpolation=Image.BICUBIC), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            self.image_transform = transforms.Compose(
                [transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def _LoadAudio(self, path, label_key):
        x, sr = sf.read(path, dtype = 'float32')
        assert sr == 16000
        length_orig = len(x)
        if length_orig > 16000 * self.audio_feat_len:
            audio_length = int(16000 * self.audio_feat_len)
            x = x[:audio_length] 
            if self.args.normalize:
                x_norm = (x - np.mean(x)) / np.std(x)
            else:
                x_norm = x
            x = torch.FloatTensor(x_norm) 
        else:
            audio_length = length_orig
            new_x = torch.zeros(int(16000 * self.audio_feat_len))
            if self.args.normalize:
                x_norm = (x - np.mean(x)) / np.std(x)
                new_x[:audio_length] = torch.FloatTensor(x_norm) 
            else:
                new_x[:audio_length] = torch.FloatTensor(x) 
            x = new_x
        return x, audio_length

    def _LoadImage(self, impath):
        img = Image.open(impath).convert('RGB')
        img = self.image_transform(img)
        return img

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """

        datum = self.data[index]
        wavpath = os.path.join(self.audio_base_path, datum['caption']['wav'])
        imgpath = os.path.join(self.image_base_path, datum['image'])
        img_id = datum['image'].split("/")[-1].split(".")[0]
        label_key = datum['caption']['wav'].split(".")[0]
        audio, nframes= self._LoadAudio(wavpath, label_key)
        img = self._LoadImage(imgpath)
        return img, audio, nframes, img_id, wavpath

    def __len__(self):
        return len(self.data)

    def collate(self, batch):
        vals = list(zip(*batch))

        collated = {}
        collated['images'] = torch.stack(vals[0])
        collated['audio'] = torch.nn.utils.rnn.pad_sequence(vals[1], batch_first=True)
        collated['audio_length'] = torch.LongTensor(vals[2])
        collated['img_id'] = np.array(vals[3])
        collated['wavpath'] = np.array(vals[4])
        collated['audio_attention_mask'] = torch.arange(len(collated['audio'][0])).unsqueeze(0) >= collated['audio_length'].unsqueeze(1)
        return collated
    