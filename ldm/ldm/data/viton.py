import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.utils.data as data
import torch
import random
import os.path
import json
import os.path as osp
from PIL import ImageDraw
import torch.utils.data as data

def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize, method)))
        osize = [256, 192]
        transform_list.append(transforms.Scale(osize, method))
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(
            lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(
            lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(
            lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

DATASET_BASE='data/viton'
class VitonDataset(data.Dataset):
    def __init__(self, phase="train", size=256, flip_p=0.5, interpolation="bicubic",use_cache=True):
        super(VitonDataset, self).__init__()
        self.phase = phase
        self.dataroot = DATASET_BASE

        self.size = size
        self.fine_height = 256
        self.fine_width = 192
        self.radius = 5

        self.label_nc = 20

        if self.phase == 'train':
            self.datapairs = 'train_pairs.txt'
        else:
            self.datapairs = 'test_pairs.txt'

        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

        self.initialize()
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        if use_cache:
            self.image_cache = {}
        else:
            self.image_cache = None

    def initialize(self):        
        # load data list from pairs file
        human_names = []
        cloth_names = []
        with open(os.path.join(self.dataroot, self.datapairs), 'r') as f:
            for line in f.readlines():
                h_name, c_name = line.strip().split()
                human_names.append(h_name)
                cloth_names.append(c_name)
        self.human_names = human_names
        self.cloth_names = cloth_names

        # input A (label maps)
        self.dir_human_parsing = os.path.join(self.dataroot, self.phase + '_label') # human paring
        self.dir_human_image = os.path.join(self.dataroot, self.phase + '_img') # human image
        self.dir_pose = os.path.join(self.dataroot, self.phase + '_pose') # predicted pose
        self.dir_cloth = os.path.join(self.dataroot, self.phase + '_color') # cloth image
        self.dir_cloth_mask = os.path.join(self.dataroot, self.phase + '_edge') # cloth mask

        # self.dir_E = os.path.join(self.dataroot, self.phase + '_edge') # predicted edge
        # self.dir_MC = os.path.join(self.dataroot, self.phase + '_colormask') # colormask dir

    def process_img(self, img_path):
        if self.image_cache is None or img_path not in self.image_cache:
            image = Image.open(img_path)
                
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

            image = (image / 127.5 - 1.0).astype(np.float32)
            self.image_cache[img_path] = image
        else:
            image = self.image_cache[img_path]
        return image
    
    def process_img_single_channel(self, img_path):
        if self.image_cache is None or img_path not in self.image_cache:
            image = Image.open(img_path)
                
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

            self.image_cache[img_path] = image
        else:
            image = self.image_cache[img_path]
        return image
    
    def __getitem__(self, index):
        train_mask = 9600
        # input A (label maps)
        box = []

        # get names from the pairs file
        c_name = self.cloth_names[index]
        h_name = self.human_names[index]
       
        # load human image
        human_image_path = osp.join(self.dir_human_image, h_name)
        human_image = self.process_img(human_image_path)
        # human_image = Image.open(human_image_path).convert('RGB')

        # load cloth image
        cloth_image_path = osp.join(self.dir_cloth, c_name)
        cloth_image = self.process_img(cloth_image_path)

        # load human parsing
        human_parsing_path = osp.join(self.dir_human_parsing, h_name.replace(".jpg", ".png"))
        human_parsing = self.process_img_single_channel(human_parsing_path)
        
        # load cloth mask
        cloth_mask_path = osp.join(self.dir_cloth_mask, c_name)
        cloth_mask = self.process_img_single_channel(cloth_mask_path)

        # Load an preprocess pose
        # pose_path = osp.join(self.dir_pose, h_name.replace('.jpg', '_keypoints.json'))
        # with open(osp.join(pose_name), 'r') as f:
        #     pose_label = json.load(f)
        #     pose_data = pose_label['people'][0]['pose_keypoints']
        #     pose_data = np.array(pose_data)
        #     pose_data = pose_data.reshape((-1, 3))
        # point_num = pose_data.shape[0]
        # pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        # r = self.radius
        # im_pose = Image.new('L', (self.fine_width, self.fine_height))
        # pose_draw = ImageDraw.Draw(im_pose)
        # for i in range(point_num):
        #     one_map = Image.new('L', (self.fine_width, self.fine_height))
        #     draw = ImageDraw.Draw(one_map)
        #     pointx = pose_data[i, 0]
        #     pointy = pose_data[i, 1]
        #     if pointx > 1 and pointy > 1:
        #         draw.rectangle((pointx-r, pointy-r, pointx +
        #                         r, pointy+r), 'white', 'white')
        #         pose_draw.rectangle(
        #             (pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
        #     one_map = transform_B(one_map.convert('RGB'))
        #     pose_map[i] = one_map[0]


        return {"img_path":human_image_path, "style_img_path":cloth_image_path, 
                "image":human_image,"style_image":cloth_image}
                # "image_mask":human_parsing, 'style_image_mask':cloth_mask}

    def __len__(self):
        return len(self.human_names)

if __name__ == "__main__":
    VitonDataset('train', 256)[0]