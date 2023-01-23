import argparse
import os
import time
import numpy as np
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt

import torch
import torch.utils.data.distributed
from tqdm import tqdm
from easydict import EasyDict as edict
import seaborn as sns; sns.set_theme()

sys.path.append('./')

from models.model_utils import create_model
from utils.misc import AverageMeter, ProgressMeter
from utils.evaluation_utils import post_processing, get_batch_statistics_rotated_bbox, ap_per_class, load_classes, post_processing_v2
from data_process_astyx.astyx_dataloader import create_val_dataloader

def evaluate_mAP(val_loader, model, configs):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    progress = ProgressMeter(len(val_loader), [batch_time, data_time],
                             prefix="Evaluation phase...")
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, batch_data in enumerate(tqdm(val_loader)):
            data_time.update(time.time() - start_time)
            _, imgs, targets = batch_data
            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale x, y, w, h of targets ((box_idx, class, x, y, w, l, im, re))
            targets[:, 2:6] *= configs.img_size
            imgs = imgs.to(configs.device, non_blocking=True)
            
            outputs = model(imgs)
            outputs = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)
            
            stats = get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold=configs.iou_thresh)
            
            sample_metrics += stats if stats else [[np.array([]), torch.tensor([]), torch.tensor([])]]
            # measure elapsed time
            # torch.cuda.synchronize()
            batch_time.update(time.time() - start_time)

            start_time = time.time()
        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


def parse_eval_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for Complex YOLO Implementation')
    
                             
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
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for class conf')
    parser.add_argument('--nms-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for nms')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for IoU')

    parser.add_argument('--plot_AP', action='store_true',
                        help='Plot the Average Precision of the first class.')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.working_dir = '../../'
    configs.dataset_dir = os.path.join(configs.working_dir, 'astyx', 'dataset', 'dataset_astyx_hires2019')
    return configs


if __name__ == '__main__':
        
    configs = parse_eval_configs()
    configs.distributed = False  # For evaluation
    class_names = load_classes(configs.classnames_infor_path)
    
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
        
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
    
    model = create_model(configs)

    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.exists(configs.pretrained_path), f'No file at {configs.pretrained_path}'

    
    model.load_state_dict(torch.load(configs.pretrained_path + checkpoint, map_location=torch.device(configs.device)))
    
    model = model.to(device=configs.device)

    model.eval()
    print('Create the validation dataloader')
    val_dataloader = create_val_dataloader(configs)

    print("\nStart computing mAP...\n")
    precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs, None)
    print("\nDone computing mAP...\n")
    for idx, cls in enumerate(ap_class):
        print("\t>>>\t Class {} ({}): precision = {:.4f}, recall = {:.4f}, AP = {:.4f}, f1: {:.4f}".format(cls, \
                class_names[cls][:3], precision[idx], recall[idx], AP[idx], f1[idx]))
    
    AP_list += [AP[0]]
    print(AP_list)
    print("\nmAP: {}\n".format(AP.mean()))
    
