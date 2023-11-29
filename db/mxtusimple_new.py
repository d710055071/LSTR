import cv2
import json
import numpy as np
import os
import pickle

import torchvision.transforms as F
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from imgaug.augmenters import Resize

from config import system_configs

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) #RGB
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class MxTUSIMPLE_New(Dataset):
    def __init__(self,db_config,split) -> None:
        super().__init__()
        # TODO 后续写配置文件里
        self._image_ids = []
        self.max_points = 0
        # 模型输入尺寸
        inp_h, inp_w = db_config['input_size']

        self.img_w, self.img_h = inp_w, inp_h

        self._image_file = []
        self.normalize = True
        self.to_tensor = F.ToTensor()
        self.aug_chance = 0.9090909090909091
        # self.aug_chance = 0.5
        self._split = split
        # 配置数据
        self._dataset = {
            # "train":['label_data_20230830'],
            # "train":['label_data_20230919', 'label_data_20230911', 'label_data_20230830'],
            "train":['label_data_train'],
            "val":['label_data_val'],
            # "val":['label_data_20230904'],
            "train+val":[],
            "test":[]
        }[self._split]
        # 构建数据路径
        self.root = os.path.join(system_configs.data_dir, 'MxTuSimple', 'LaneDetection')
        if self.root is None:
            raise Exception('Please specify the root directory')
        # 构建标签文件路径集
        self.anno_files = [os.path.join(self.root, path + '.json') for path in self._dataset]
        # 缓存数据路径
        self._cache_file = os.path.join(system_configs.cache_dir, "tusimple_{}.pkl".format(self._dataset))

        transformations = iaa.Sequential([Resize({'height': inp_h, 'width': inp_w})])


        if self._split not in ["val","test"]:
            # 数据增强
            self.augmentations = [
                                    {'name': 'Affine', 'parameters': {'rotate': (-10, 10)}},
                                    {'name': 'HorizontalFlip', 'parameters': {'p': 0.5}},
                                    # {'name': 'CropToFixedSize', 'parameters': {'height': 1080, 'width': 1920}}
                                ]

            augmentations = [getattr(iaa, aug['name'])(**aug['parameters'])
                                for aug in self.augmentations]  # add augmentation
            self.transform = iaa.Sequential([iaa.Sometimes(then_list=augmentations, p=self.aug_chance), transformations])
        else:
            self.transform = iaa.Sequential([transformations])
        # 加载数据
        self._load_data()




        
        # self.need_transforms = True
        # if self._split in ["val","test"]:
        #     self.need_transforms = False
        self.ColorJitter = F.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
    def _load_data(self):
        print("loading from cache file: {}".format(self._cache_file))
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            self._extract_data()
            self._transform_annotations()

            with open(self._cache_file, "wb") as f:
                pickle.dump([self._annotations,
                             self._image_ids,
                             self._image_file,
                             self.max_lanes,
                             self.max_points], f)
        else:
            with open(self._cache_file, "rb") as f:
                (self._annotations,
                 self._image_ids,
                 self._image_file,
                 self.max_lanes,
                 self.max_points) = pickle.load(f)
    def _extract_data(self):

        max_lanes = 0
        image_id  = 0

        self._old_annotations = {}

        for anno_file in self.anno_files:
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line) # lanes list h_sample list raw_file str
                if 'categories' in data:
                    categories = data['categories']
                    # categories =  [1] * len(data['lanes'])
                y_samples = data['h_samples']
                gt_lanes = data['lanes']  # 4 lanes
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
                lanes = [lane for lane in lanes if len(lane) > 0]
                max_lanes = max(max_lanes, len(lanes))
                self.max_lanes = max_lanes
                self.max_points = max(self.max_points, max([len(l) for l in gt_lanes]))
                img_path  = os.path.join(self.root, data['raw_file'])
                self._image_file.append(img_path)
                self._image_ids.append(image_id)
                self._old_annotations[image_id] = {
                    'path': img_path,
                    'org_path': data['raw_file'],
                    'org_lanes': gt_lanes,
                    'lanes': lanes,
                    'aug': False,
                    'y_samples': y_samples,
                    'categories':categories
                }
                image_id += 1

    def _transform_annotations(self):
        print('Now transforming annotations...')
        self._annotations = {}
        for image_id, old_anno in self._old_annotations.items():
            self._annotations[image_id] = self._transform_annotation(old_anno)

    def _transform_annotation(self, anno, img_wh=None):
        if img_wh is None:
            img_h = self._get_img_heigth(anno['path'])
            img_w = self._get_img_width(anno['path'])
        else:
            img_w, img_h = img_wh

        old_lanes = anno['lanes']
        categories = anno['categories'] if 'categories' in anno else [1] * len(old_lanes)
        old_lanes = zip(old_lanes, categories)
        old_lanes = filter(lambda x: len(x[0]) > 0, old_lanes)
        lanes = np.ones((self.max_lanes, 1 + 2 + 2 * self.max_points), dtype=np.float32) * -1e5
        lanes[:, 0] = 0
        old_lanes = sorted(old_lanes, key=lambda x: x[0][0][0])
        for lane_pos, (lane, category) in enumerate(old_lanes):
            lower, upper = lane[0][1], lane[-1][1]
            xs = np.array([p[0] for p in lane]) / img_w
            ys = np.array([p[1] for p in lane]) / img_h
            lanes[lane_pos, 0] = category
            lanes[lane_pos, 1] = lower / img_h
            lanes[lane_pos, 2] = upper / img_h
            lanes[lane_pos, 3:3 + len(xs)] = xs
            lanes[lane_pos, (3 + self.max_points):(3 + self.max_points + len(ys))] = ys

        new_anno = {
            'path': anno['path'],
            'label': lanes,
            'old_anno': anno,
            'categories': [cat for _, cat in old_lanes]
        }
        return new_anno

    def _get_img_heigth(self, path):
        return 1080

    def _get_img_width(self, path):
        return 1920

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)
        return lanes

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))
        return lines
    
    def __len__(self):
        return len(self._annotations)
    
    def __getitem__(self, idx):
        image_id = self._image_ids[idx]
        item = self._annotations[image_id]
        
        img = cv2.imread(item['path'])
        # try :
        mask  = np.ones((1, img.shape[0], img.shape[1], 1), dtype=np.bool)
        # except Exception as e :
        #     print(item['path'])
        # finally :
        #     print('最后执行')
        # BGR->RGB
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        label = item['label']
        # 数据增强需要重新制作label
        # if self.need_transforms:

        line_strings = self.lane_to_linestrings(item['old_anno']['lanes'])
        line_strings = LineStringsOnImage(line_strings, shape=img.shape)
        img, line_strings,mask = self.transform(image=img, line_strings=line_strings,segmentation_maps=mask)
        line_strings.clip_out_of_image_()
        new_anno = {'path': item['path'], 'lanes': self.linestrings_to_lanes(line_strings)}
        new_anno['categories'] = item['old_anno']['categories']
        label = self._transform_annotation(new_anno, img_wh=(self.img_w, self.img_h))['label']
        
        img = img / 255.
        # if self.normalize:
        #     img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = self.to_tensor(img.astype(np.float32))
        # if self.need_transforms:
        #     img = self.ColorJitter(img)

        mask = np.logical_not(mask[:, :, :, 0]).astype(np.float32)
        return (img, label, mask, idx,item["path"])