# SRC
## Training

```shell script
python train.py \
  --saved_fn 'complex_yolov4' \
  --arch 'darknet' \
  --cfgfile ./config/cfg/complex_yolov4.cfg \
  --batch_size 8 \
  --num_workers 4 \
  --gpu_idx 0 \
  --radar\
  --mag \
  --no-val 
```
Traines using only RADAR data with additonal information magnitude. Here gpu 0 will be used, if you want to use the cpu, erase the parameter gpu_idx 0 and add the flag no_cuda.

## Evaluation

```shell script
python evaluate.py \
  --arch 'darknet' \
  --cfgfile './config/cfg/complex_yolov4.cfg'\
  --batch_size 1 \
  --num_workers 1 \
  --pretrained_path '../checkpoints/low_fusion_mag.pth' \
  --img_size 608 \
  --conf_thresh 0.5 \
  --nms_thresh 0.5 \
  --gpu_idx 0 \
  --low_fusion \
  --mag \
```
Evaluates the network using low fusion data.
