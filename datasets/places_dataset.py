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
        parser.add_argument("--train_audio_dataset_json_file", type=str, default="/places_root/metadata/train_2020.json")
        parser.add_argument("--val_seen_audio_dataset_json_file", type=str, default="/places_root/metadata/dev_seen_2020.json")
        parser.add_argument("--val_unseen_audio_dataset_json_file", type=str, default="/places_root/metadata/dev_unseen_2020.json")
        parser.add_argument("--audio_feat_len", type=float, help="maximal audio length", default=18)
        parser.add_argument("--val_audio_feat_len", type=float, help="maximal audio length", default=25.)
        parser.add_argument("--normalize", action="store_true", default=False, help="whether or not normalize raw input, both w2v2 and hubert base doesn't normalize the input, but in exps in two papers, we normalized it, hopefully this doesn't make a difference")


    def __init__(self, args, split = "train"):
        self.args = args
        self.split = split
        self.audio_feat_len = args.audio_feat_len if "train" in split else args.val_audio_feat_len
        if split == "train":
            json_path = args.train_audio_dataset_json_file
        elif split == "val_seen":
            json_path = args.val_seen_audio_dataset_json_file
        elif split == "val_unseen":
            json_path = args.val_unseen_audio_dataset_json_file
        else:
            raise NotImplementedError
        raise RuntimeError(f"please replace /places_root/ with the actual root where you download the dataset, after than please delete this line")
        self.image_base_path = "/places_root/PlacesAudio_400k_distro/images"
        self.audio_base_path = "/places_root/PlacesAudio_400k_distro"
        
        with open(json_path, 'r') as f:
            json_file = json.load(f)

        self.data = json_file["data"]
        
        if "train" not in split:
            self.image_transform = transforms.Compose(
                [transforms.Resize(256, interpolation=Image.BICUBIC), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            self.image_transform = transforms.Compose(
                [transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def _LoadAudio(self, path):
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
            if self.args.normalize:
                x_norm = (x - np.mean(x)) / np.std(x)
            else:
                x_norm = x
            new_x = torch.zeros(int(16000 * self.audio_feat_len))
            new_x[:audio_length] = torch.FloatTensor(x_norm) 
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
        wavpath = os.path.join(self.audio_base_path, datum['wav'])
        imgpath = os.path.join(self.image_base_path, datum['image'])
        audio, nframes = self._LoadAudio(wavpath)
        img = self._LoadImage(imgpath)
        return img, audio, nframes, imgpath, wavpath

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

