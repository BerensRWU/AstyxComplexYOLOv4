# AstyxComplexYOLOv4
This repository contains a PyTorch implementation of [ComplexYOLO](https://arxiv.org/pdf/1803.06199.pdf) using YOLO version 4. For an implementation using YOLOv3 see [here](https://github.com/BerensRWU). It is build to be applied on the data from the Astyx Dataset. For an implementation for the KITTI dataset see [here](https://github.com/maudzung/Complex-YOLOv4-Pytorch).

## Requirement

```shell script
pip install -U -r requirements.txt
```

#### Steps
1. Install all requirements
1. Download or clone this repo by using ```git clone https://github.com/BerensRWU/AstyxComplexYOLOv4/``` in the terminal.
1. Save the Astyx dataset in the folder ```dataset```.(See Section Astyx HiRes).
1. Download the weights for the RADAR and LiDAR detector from the moodle page of the Lecture. 

# Astyx HiRes
The Astyx HiRes is a dataset from Astyx for object detection for autonomous driving. Astyx has a sensor setup consisting of camera, LiDAR, RADAR. Additional information can be found here: [Dataset Paper](https://www.astyx.com/fileadmin/redakteur/dokumente/Automotive_Radar_Dataset_for_Deep_learning_Based_3D_Object_Detection.PDF) and [Specification](https://www.astyx.com/fileadmin/redakteur/dokumente/Astyx_Dataset_HiRes2019_specification.pdf)

```
└── dataset/
       ├── dataset_astyx_hires2019    <-- 546 data
       |   ├── calibration 
       |   ├── camera_front
       |   ├── groundtruth_obj3d
       |   ├── lidar_vlp16
       └── ├── radar_6455 
```
