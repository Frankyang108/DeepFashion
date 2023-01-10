import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.utils.data as data
import torch
import random

DATASET_BASE='/data/DeepFashion/DeepFashion'
class FashionInshop(Dataset):
    def __init__(self, type="train", size=256, flip_p=0.5, interpolation="bicubic", transform=None):
        self.transform = transform
        self.type = type
        self.train_dict = {}
        self.test_dict = {}
        self.train_list = []
        self.test_list = []
        self.all_path = []
        self.cloth = self.readcloth()
        self.read_train_test()
        self.size = size
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

    def read_lines(self, path):
        with open(path) as fin:
            lines = fin.readlines()[2:]
            lines = list(filter(lambda x: len(x) > 0, lines))
            pairs = list(map(lambda x: x.strip().split(), lines))
        return pairs

    def readcloth(self):
        lines = self.read_lines(os.path.join(DATASET_BASE, 'Anno', 'list_bbox_inshop.txt'))
        valid_lines = list(filter(lambda x: x[1] == '1', lines))
        names = set(list(map(lambda x: x[0], valid_lines)))
        return names

    def read_train_test(self):
        lines = self.read_lines(os.path.join(DATASET_BASE, 'Eval', 'list_eval_partition.txt'))
        valid_lines = list(filter(lambda x: x[0] in self.cloth, lines))
        for line in valid_lines:
            s = self.train_dict if line[2] == 'train' else self.test_dict
            if line[1] not in s:
                s[line[1]] = [line[0]]
            else:
                s[line[1]].append(line[0])

        def clear_single(d):
            keys_to_delete = []
            for k, v in d.items():
                if len(v) < 2:
                    keys_to_delete.append(k)
            for k in keys_to_delete:
                d.pop(k, None)
        clear_single(self.train_dict)
        clear_single(self.test_dict)
        self.train_list, self.test_list = list(self.train_dict.keys()), list(self.test_dict.keys())
        for v in list(self.train_dict.values()):
            self.all_path += v
        self.train_len = len(self.all_path)
        for v in list(self.test_dict.values()):
            self.all_path += v
        self.test_len = len(self.all_path) - self.train_len

    def process_img(self, img_path):
        image = Image.open(os.path.join(DATASET_BASE, 'Img', img_path))
            
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
            (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)

        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)

        return (image / 127.5 - 1.0).astype(np.float32)

    def __len__(self):
        if self.type == 'test':
            return len(self.test_list)
        else:
            return len(self.all_path)

    def __getitem__(self, item):
        if self.type == 'all':
            img_path = self.all_path[item]
            return {"file_path_":img_path, "image":self.process_img(img_path)}
        
        if self.type == 'train':
            item = item % len(self.train_list)
        elif self.type == 'test':
            item = item % len(self.test_list)

        example = {}
        s_d = self.train_dict if self.type == 'train' else self.test_dict
        s_l = self.train_list if self.type == 'train' else self.test_list
        imgs = s_d[s_l[item]]
        img_path_pairs = random.sample(imgs, 2)
        img_pairs = list(map(self.process_img, img_path_pairs))
        return {"img_path":img_path_pairs[0], "style_img_path":img_path_pairs[1], "image":img_pairs[0],"style_image":img_pairs[1]}

class LSUNBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example