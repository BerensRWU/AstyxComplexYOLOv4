"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.08
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: Testing script
"""

import argparse
import sys
import os
import time

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np

sys.path.append('../')

import config.astyx_config as cnf
from models.model_utils import create_model
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import post_processing, rescale_boxes, post_processing_v2

from data_process_astyx.astyx_dataloader import create_test_dataloader
from data_process_astyx import astyx_data_utils as data_utils
from data_process_astyx import astyx_bev_utils as bev_utils

def parse_test_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for Complex YOLO Implementation')
    parser.add_argument('--saved_fn', type=str, default='complexer_yolov4', metavar='FN',
                        help='The name using for saving logs, models,...')
    
    parser.add_argument('--working-dir', type=str, default='../', metavar='PATH',
                        help='The ROOT working directory')
    
    
    parser.add_argument('--lidar', action='store_true',
                        help='Use LiDAR data instead of RADAR.')
    parser.add_argument('--radar', action='store_true',
                        help='Use RADAR data instead of LiDAR.')
    parser.add_argument('--VR', action='store_true',
                        help='Use the radial velocity from the RADAR data.')
    parser.add_argument('--mag', action='store_true',
                        help='Use the magnitude from the RADAR data.')
    parser.add_argument('--low_fusion', action='store_true',
                        help='Low Level Fusion using RADAR and LiDAR data.')
                        
                        
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='./config/cfg/complex_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')

    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=None, type=int,
                        help='GPU index to use.')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf_thresh', type=float, default=0.5,
                        help='the threshold for conf')
    parser.add_argument('--nms_thresh', type=float, default=0.5,
                        help='the threshold for conf')

    parser.add_argument('--show_image', action='store_true',
                        help='If true, show the image during demostration')
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_complexer_yolov4', metavar='PATH',
                        help='the video filename if the output format is video')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    
    if configs.no_cuda:
        configs.device = "cpu"
    else:
        configs.device = "cuda"
    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    if configs.low_fusion:
        sensor = 'low_fusion'
    elif configs.radar:
        sensor = "radar"
    elif configs.lidar:
        sensor = 'lidar'
    else:
        raise NotImplementedError
    
    if sensor != 'lidar':
        if configs.VR:
            sensor += "_VR"
        elif configs.mag:
            sensor += "_Mag"
        else:
            raise NotImplementedError
    
    configs.saved_fn = f'{configs.saved_fn}_{configs.dataset}_{sensor}'
    
    configs.dataset_dir = os.path.join(configs.working_dir, '../', 'astyx', 'dataset', 'dataset_astyx_hires2019')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.working_dir, 'results', configs.dataset, configs.saved_fn)
        make_folder(configs.results_dir)

    return configs


if __name__ == '__main__':
    configs = parse_test_configs()

    model = create_model(configs)

    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location=torch.device(configs.device)))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)

    out_cap = None

    model.eval()

    test_dataloader = create_test_dataloader(configs)
    with torch.no_grad():
        for batch_idx, (img_paths, imgs_bev,targets) in enumerate(test_dataloader):
            if batch_idx != 44:
                continue
            input_imgs = imgs_bev.to(device=configs.device).float()
            t1 = time_synchronized()
            outputs = model(input_imgs)
            t2 = time_synchronized()
            detections = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)

            img_detections = []  # Stores detections for each image index
            img_detections.extend(detections)

            img_bev = imgs_bev.squeeze() * 255
            img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
            img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))
            print(img_bev.shape)
            img_bev[1:] = np.max(np.concatenate([img_bev[1:], img_bev[:-1]]).reshape(2,configs.img_size-1, configs.img_size,3),0)

            img_bev[2:] = np.max(np.concatenate([img_bev[2:], img_bev[:-2]]).reshape(2,configs.img_size-2, configs.img_size,3),0)

            img_bev[:,1:] = np.max(np.concatenate([img_bev[:,1:], img_bev[:,:-1]]).reshape(2,configs.img_size, configs.img_size-1,3),0)
            img_bev[:,2:] = np.max(np.concatenate([img_bev[:,2:], img_bev[:,:-2]]).reshape(2,configs.img_size, configs.img_size-2,3),0)
            print(img_bev.shape)

            targets = targets.reshape((-1,8))
            
            targets[:, 2:6] *= configs.img_size
            for targets_ in targets:

                if targets_ is None:
                    continue
                
                _, cls_pred, x, y, w, l, im, re = targets_
                yaw = np.arctan2(im, re)
                # Draw rotated box
                bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, [255,255,255])

            for detections in img_detections:
                if detections is None:
                    continue
                # Rescale boxes to original image
                detections = rescale_boxes(detections, configs.img_size, img_bev.shape[:2])
                
                for x, y, w, l, im, re, *_, cls_pred in detections:
                    yaw = np.arctan2(im, re)
                    # Draw rotated box
                    bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, [100,255,255])
            img_rgb = cv2.imread(img_paths[0])
                
            img_bev = cv2.flip(cv2.flip(img_bev, 0), 1)
            out_img = img_bev

            print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS'.format(batch_idx, (t2 - t1) * 1000,
                                                                                           1 / (t2 - t1)))

            if configs.save_test_output:
                if configs.output_format == 'image':
                    img_fn = os.path.basename(img_paths[0])[:-4]
                    cv2.imwrite(os.path.join(configs.results_dir, '{}.jpg'.format(img_fn)), out_img[:,:,[2,1,0]])
                elif configs.output_format == 'video':
                    if out_cap is None:
                        out_cap_h, out_cap_w = out_img.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        out_cap = cv2.VideoWriter(
                            os.path.join(configs.results_dir, '{}.avi'.format(configs.output_video_fn)),
                            fourcc, 30, (out_cap_w, out_cap_h))

                    out_cap.write(out_img)
                else:
                    raise TypeError

            if configs.show_image:
                cv2.imshow('test-img', out_img)
                print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
                if cv2.waitKey(0) & 0xFF == 27:
                    break
    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()
