"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.05
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for the KITTI dataset

# Refer: https://github.com/ghimiredhikura/Complex-YOLOv3
"""

import sys
import os
import random

import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import cv2

sys.path.append('../')

from data_process_astyx import transformation, astyx_bev_utils, astyx_data_utils
import config.kitti_config as cnf


class astyxDataset(Dataset):
    def __init__(self, dataset_dir, mode='train', point_cloud_transforms=None, 
                 aug_transforms=None, multiscale=False, num_samples=None,
                 configs=None):

        self.dataset_dir = dataset_dir
        assert mode in ['train', 'valid', 'test'], f'Invalid mode: {mode}'
        self.mode = mode
        self.is_test = (self.mode == 'test')

        self.radar = configs.radar
        self.lidar = configs.lidar
        self.low_fusion = configs.low_fusion
        self.VR = configs.VR
        self.mag = configs.mag
        self.multiscale = multiscale
        self.point_cloud_transforms = point_cloud_transforms
        self.aug_transforms = aug_transforms
        self.img_size = cnf.BEV_WIDTH
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        
        self.lidar_dir = os.path.join(self.dataset_dir, 'lidar_vlp16')
        self.radar_dir = os.path.join(self.dataset_dir, 'radar_6455')
        self.image_dir = os.path.join(self.dataset_dir, 'camera_front')
        self.calib_dir = os.path.join(self.dataset_dir, 'calibration')
        self.label_dir = os.path.join(self.dataset_dir, 'groundtruth_obj3d')

        if self.mode == "train":
            self.sample_id_list = [*list(range(39)),*list(range(130,546))]
        else:
            self.sample_id_list = [*list(range(39,130))]
                
        if num_samples is not None:
            self.sample_id_list = self.sample_id_list[:num_samples]
        self.num_samples = len(self.sample_id_list)

    def __getitem__(self, index):
        return self.load_img_with_targets(index)

    def load_img_with_targets(self, index):
        """Load images and targets for the training and validation phase"""

        sample_id = int(self.sample_id_list[index])

        
        #lidarData = self.get_lidar(sample_id)
        objects = self.get_label(sample_id)
        calib = self.get_calib(sample_id)
        
        labels, noObjectLabels = astyx_bev_utils.read_labels_for_bevbox(objects)

        target = astyx_bev_utils.build_yolo_target(labels)
        img_file = os.path.join(self.image_dir, '{:06d}.jpg'.format(sample_id))

        # on image space: targets are formatted as (box_idx, class, x, y, w, l, im, re)
        n_target = len(target)
        targets = torch.zeros((n_target, 8))
        if n_target > 0:
            targets[:, 1:] = torch.from_numpy(target)
                    
        if self.low_fusion:
                
            pcData = self.get_lidar(sample_id)
            intensity = pcData[:,3].reshape(-1,1)
            pcData = calib.lidar2ref(pcData[:,0:3])
            pcData = np.concatenate([pcData,intensity],1)
            pcData = np.concatenate([pcData,self.get_radar(sample_id)])
            
                
        elif self.radar:
            pcData = self.get_radar(sample_id)
            
        elif self.lidar:
            pcData = self.get_lidar(sample_id)
            intensity = pcData[:,3].reshape(-1,1)
            pcData = calib.lidar2ref(pcData[:,0:3])
            pcData = np.concatenate([pcData,intensity],1)
        else:
            raise NotImplementedError

    
        if self.point_cloud_transforms is not None:
            pcData, labels[:, 1:] = self.point_cloud_transforms(pcData, labels[:, 1:])
            
        b = astyx_bev_utils.removePoints(pcData, cnf.boundary)
        rgb_map = astyx_bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)

        rgb_map = torch.from_numpy(rgb_map).float()

        if self.aug_transforms is not None:
            rgb_map, targets = self.aug_transforms(rgb_map, targets)

        return img_file, rgb_map, targets

    def __len__(self):
        return len(self.sample_id_list)

    def check_point_cloud_range(self, xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range = [cnf.boundary["minX"], cnf.boundary["maxX"]]
        y_range = [cnf.boundary["minY"], cnf.boundary["maxY"]]
        z_range = [cnf.boundary["minZ"], cnf.boundary["maxZ"]]

        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

    def collate_fn(self, batch):
   
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if (self.batch_count % 10 == 0) and self.multiscale:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack(imgs)
        if self.img_size != cnf.BEV_WIDTH:
            imgs = F.interpolate(imgs, size=self.img_size, mode="bilinear", align_corners=True)
        self.batch_count += 1

        return paths, imgs, targets

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, f'{idx:06d}.jpg')
        assert os.path.isfile(img_file)
        return cv2.imread(img_file)  # (H, W, C) -> (H, W, 3) OpenCV reads in BGR mode

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, f'{idx:06d}.txt')
        assert os.path.isfile(lidar_file)
        lidar = np.loadtxt(lidar_file, dtype=np.float32, skiprows = 1)
        lidar = lidar[:,0:4]

        return lidar
        
    def get_radar(self, idx):
        radar_file = os.path.join(self.radar_dir, f'{idx:06d}.txt')
        assert os.path.isfile(radar_file)
        radar = np.loadtxt(radar_file, dtype=np.float32, skiprows = 2)
        if self.VR:
            radar = radar[:,[0,1,2,3]]
        elif self.mag:
            radar = radar[:,[0,1,2,4]]
        else:
            raise NotImplementedError

        return radar

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, f'{idx:06d}.json')
        assert os.path.isfile(calib_file)
        return astyx_data_utils.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, f'{idx:06d}.json'.format(idx))
        assert os.path.isfile(label_file)
        return astyx_data_utils.read_label(label_file)
