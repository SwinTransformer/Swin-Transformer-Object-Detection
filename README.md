# Swin Transformer for Object Detection

This repo contains the supported code and configuration files to reproduce object detection results of [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf). It is based on [mmdetection](https://github.com/open-mmlab/mmdetection).

## Updates

***05/11/2021*** Models for [MoBY](https://github.com/SwinTransformer/Transformer-SSL) are released

***04/12/2021*** Initial commits

## Results and Models

### Mask R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T | ImageNet-1K | 1x | 43.7 | 39.8 | 48M | 267G | [config](configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/mask_rcnn_swin_tiny_patch4_window7_1x.log.json)/[baidu](https://pan.baidu.com/s/1bYZk7BIeFEozjRNUesxVWg) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/mask_rcnn_swin_tiny_patch4_window7_1x.pth)/[baidu](https://pan.baidu.com/s/19UOW0xl0qc-pXQ59aFKU5w) |
| Swin-T | ImageNet-1K | 3x | 46.0 | 41.6 | 48M | 267G | [config](configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/mask_rcnn_swin_tiny_patch4_window7.log.json)/[baidu](https://pan.baidu.com/s/1Te-Ovk4yaavmE4jcIOPAaw) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/mask_rcnn_swin_tiny_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1YpauXYAFOohyMi3Vkb6DBg) |
| Swin-S | ImageNet-1K | 3x | 48.5 | 43.3 | 69M | 359G | [config](configs/swin/mask_rcnn_swin_small_patch4_window7_mstrain_480-800_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/mask_rcnn_swin_small_patch4_window7.log.json)/[baidu](https://pan.baidu.com/s/1ymCK7378QS91yWlxHMf1yw) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/mask_rcnn_swin_small_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1V4w4aaV7HSjXNFTOSA6v6w) |

### Cascade Mask R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T | ImageNet-1K | 1x | 48.1 | 41.7 | 86M | 745G | [config](configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/cascade_mask_rcnn_swin_tiny_patch4_window7_1x.log.json)/[baidu](https://pan.baidu.com/s/1x4vnorYZfISr-d_VUSVQCA) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/cascade_mask_rcnn_swin_tiny_patch4_window7_1x.pth)/[baidu](https://pan.baidu.com/s/1vFwbN1iamrtwnQSxMIW4BA) |
| Swin-T | ImageNet-1K | 3x | 50.4 | 43.7 | 86M | 745G | [config](configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_tiny_patch4_window7.log.json)/[baidu](https://pan.baidu.com/s/1GW_ic617Ak_NpRayOqPSOA) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_tiny_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1i-izBrODgQmMwTv6F6-x3A) |
| Swin-S | ImageNet-1K | 3x | 51.9 | 45.0 | 107M | 838G | [config](configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_small_patch4_window7.log.json)/[baidu](https://pan.baidu.com/s/17Vyufk85vyocxrBT1AbavQ) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_small_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1Sv9-gP1Qpl6SGOF6DBhUbw) |
| Swin-B | ImageNet-1K | 3x | 51.9 | 45.0 | 145M | 982G | [config](configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_base_patch4_window7.log.json)/[baidu](https://pan.baidu.com/s/1UZAR39g-0kE_aGrINwfVHg) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_base_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1tHoC9PMVnldQUAfcF6FT3A) |

### RepPoints V2

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T | ImageNet-1K | 3x | 50.0 | - | 45M | 283G | [config](configs/swin/reppoitsv2_swin_tiny_patch4_window7_mstrain_480_960_giou_gfocal_bifpn_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.9/reppointsv2_swin_tiny_patch4_window7_3x.log.json) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.9/reppointsv2_swin_tiny_patch4_window7_3x.pth) |

### Mask RepPoints V2

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T | ImageNet-1K | 3x | 50.4 | 43.8 | 47M | 292G | [config](configs/swin/mask_reppoitsv2_swin_tiny_patch4_window7_mstrain_480_960_giou_gfocal_bifpn_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.9/mask_reppointsv2_swin_tiny_patch4_window7_3x.log.json) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.9/mask_reppointsv2_swin_tiny_patch4_window7_3x.pth) |

**Notes**: 

- **Pre-trained models can be downloaded from [Swin Transformer for ImageNet Classification](https://github.com/microsoft/Swin-Transformer)**.
- Access code for `baidu` is `swin`.

## Results of MoBY with Swin Transformer

### Mask R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T | ImageNet-1K | 1x | 43.6 | 39.6 | 48M | 267G | [config](configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_mask_rcnn_swin_tiny_patch4_window7_1x.log.json)/[baidu](https://pan.baidu.com/s/1P5gCIfLUQ64jbVMOom0H3w) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_mask_rcnn_swin_tiny_patch4_window7_1x.pth)/[baidu](https://pan.baidu.com/s/1xGRihuIrGVreFKn5eJ6oTg) |
| Swin-T | ImageNet-1K | 3x | 46.0 | 41.7 | 48M | 267G | [config](configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_mask_rcnn_swin_tiny_patch4_window7_3x.log.json)/[baidu](https://pan.baidu.com/s/17WAhUmhAam1of3hXOu-wtA) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_mask_rcnn_swin_tiny_patch4_window7_3x.pth)/[baidu](https://pan.baidu.com/s/1MSj8cC1wlQU1QaXCdKrzeA) |

### Cascade Mask R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T | ImageNet-1K | 1x | 48.1 | 41.5 | 86M | 745G | [config](configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_cascade_mask_rcnn_swin_tiny_patch4_window7_1x.log.json)/[baidu](https://pan.baidu.com/s/1eOdq1rvi0QoXjc7COgiM7A) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_cascade_mask_rcnn_swin_tiny_patch4_window7_1x.pth)/[baidu](https://pan.baidu.com/s/1-gbY-LExbf0FgYxWWs8OPg) |
| Swin-T | ImageNet-1K | 3x | 50.2 | 43.5 | 86M | 745G | [config](configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_cascade_mask_rcnn_swin_tiny_patch4_window7_3x.log.json)/[baidu](https://pan.baidu.com/s/1zEFXHYjEiXUCWF1U7HR5Zg) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_cascade_mask_rcnn_swin_tiny_patch4_window7_3x.pth)/[baidu](https://pan.baidu.com/s/1FMmW0GOpT4MKsKUrkJRgeg) |

**Notes:**

- The drop path rate needs to be tuned for best practice.
- MoBY pre-trained models can be downloaded from [MoBY with Swin Transformer](https://github.com/SwinTransformer/Transformer-SSL).

## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation and dataset preparation.

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

### Training

To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
For example, to train a Cascade Mask R-CNN model with a `Swin-T` backbone and 8 gpus, run:
```
tools/dist_train.sh configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 8 --cfg-options model.pretrained=<PRETRAIN_MODEL> 
```

**Note:** `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.


### Apex (optional):
We use apex for mixed precision training by default. To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the [configuration files](configs/swin):
```
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

## Citing Swin Transformer
```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

## Other Links

> **Image Classification**: See [Swin Transformer for Image Classification](https://github.com/microsoft/Swin-Transformer).

> **Semantic Segmentation**: See [Swin Transformer for Semantic Segmentation](https://github.com/Transformer/Swin-Transformer-Semantic-Segmentation).

> **Self-Supervised Learning**: See [MoBY with Swin Transformer](https://github.com/SwinTransformer/Transformer-SSL).

> **Video Recognition**, See [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer).

## swin-T moe 

I added Swin Transformer MoE (referred to as Swin-T MoE hereafter) to the backbone network. MoE is a method that expands the model parameters and improves the model performance. The implementation of Swin Transformer MoE used Microsoft's Tutel framework.

### Install [Tutel](https://github.com/microsoft/tutel)

```
python3 -m pip uninstall tutel -y 
python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@main
```

You can check out Swin-T MoE at .

```
.\mmdet\models\backbones\swin_transformer_moe.py.
```

I provided the relevant configuration files for reference:

contains the parameters for the Swin-T MoE backbone network:

```
.\configs\swin\cascade_mask_rcnn_swin_moe_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py
```

contains the modified configuration for the backbone network:

```
.\configs\swin\cascade_mask_rcnn_swin_moe_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py
```

As the output of Swin-T MoE is different from Swin-T, I modified the `extract_feat` function in `.\mmdet\models\detectors\two_stage.py`.

You can change the config according to your needs.
