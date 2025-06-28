# RepMobile
About the repo for paper: RepMobile: A Deep Mobile VGG-style CNN with Better Identity Mapping

## **News**
2025/06/30 Code is open source.


## **Abstract**

VGG-style architecture RepVGG demonstrates the effectiveness of re-parameterization techniques, which re-parameterize residual connections and convolutions to optimize the trade-off between accuracy and inference speed. 
However, the largest RepVGG variant is restricted to 28 layers, making it shallower than lightweight models like the 53-layer MobileNetV2 and consequently limiting its representational capacity. 
Subsequent MetaFormer-style reparameterizable architectures like FastViT and RepViT integrate Feed-Forward Networks to enable deeper structures but reintroduce non-reparameterizable residual connections, directly influencing the inference speed.
Our analysis of ResNet and RepVGG points out a key limitation in current VGG-style models: imperfect identity mapping impedes effective training in deeper configurations. 
To address this, we investigate strategies for better preserving identity mapping in efficient reparameterizable networks, proposing three design principles centered on normalization, downsampling, and the activation function. 
By systematically adapting a classical MobileNet architecture to adhere to these principles, we introduce RepMobile, a novel deep mobile VGG-style Convolutional Neural Network (CNN). 
Extensive experiments show that RepMobile achieves performance comparable to state-of-the-art lightweight CNNs and Vision Transformers. 
Specifically, RepMobile-S outperforms EfficientFormerV2-S0 by 1.5\% in Top-1 accuracy while delivering approximately 7X faster inference. 
Moreover, RepMobile-M runs 10X faster than RepViT-M0.9 while maintaining competitive accuracy. 

## Getting Started
The steps to create env, train and evaluate RepMobile models：

```
conda create -n RepMobile python=3.10 
conda activate RepMobile
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install -U openmim
mim install mmcv-full
mim install mmdet
mim install mmseg
pip install thop
pip install coremltools==6.3
pip install prettytable
```

## Prepare datasets

If your folder structure is different, you may need to change the corresponding paths in config files.

**For ImageNet data**, ImageNet is an image database organized according to the WordNet hierarchy. Download and extract ImageNet train and val images from http://image-net.org/. It is recommended to symlink the dataset root to `$RepMobile/data`. Organize the data into the following directory structure:
```
imagenet/
├── train/
│   ├── n01440764/  (Example synset ID)
│   │   ├── image1.JPEG
│   │   ├── image2.JPEG
│   │   └── ...
│   ├── n01443537/  (Another synset ID)
│   │   └── ...
│   └── ...
└── val/
    ├── n01440764/  (Example synset ID)
    │   ├── image1.JPEG
    │   └── ...
    └── ...
```

**For COCO data**, prepare COCO 2017 dataset according to the [instructions in MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#test-existing-models-on-standard-datasets).
The dataset should be organized as: 
```
detection
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

**For ADE20K data**, ADE20K dataset can be downloaded and prepared following [insructions in MMSeg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets). 
The data should appear as: 
```
├── segmentation
│   ├── data
│   │   ├── ade
│   │   │   ├── ADEChallengeData2016
│   │   │   │   ├── annotations
│   │   │   │   │   ├── training
│   │   │   │   │   ├── validation
│   │   │   │   ├── images
│   │   │   │   │   ├── training
│   │   │   │   │   ├── validation

```

## Model Training and Inference

### **Classification on ImageNet-1K**

```
###Trainig
# sh dist_train.sh XXX RepMobile_M 23426 4 RepMobile_M 0.0 0.0 
# sh dist_train.sh XXX RepMobile_S 23427 4 RepMobile_S 0.0 0.0 
# sh dist_train.sh XXX RepMobile_L 20427 4 RepMobile_L 0.8 1.0

###Testing
#sh dist_test.sh XXX RepMobile_M 23426 4 RepMobile_M pretrain_model/RepMobile_M.pth 
#sh dist_test.sh XXX RepMobile_S 23426 4 RepMobile_S pretrain_model/RepMobile_S.pth  
#sh dist_test.sh XXX RepMobile_L 23426 4 RepMobile_L pretrain_model/RepMobile_L.pth 

```

### **Object Detection and Instance Segmentation on COCO**

```
cd detection
###Trainig
# sh dist_train.sh XXX coct2 configs/mask_rcnn_repmobile_small_fpn_1x_coco.py 8 20234 'work_dirs/mask_rcnn_repmobile_small_fpn_1x_coco/latest.pth'
# sh dist_train.sh XXX coct1 configs/mask_rcnn_repmobile_large_fpn_1x_coco.py 8 20236 'work_dirs/mask_rcnn_repmobile_large_fpn_1x_coco/latest.pth' 

###Testing
# sh dist_test.sh XXX coct1 20345 configs/mask_rcnn_repmobile_small_fpn_1x_coco.py 'work_dirs/mask_rcnn_repmobile_large_fpn_1x_coco/epoch_12.pth' 1  
# sh dist_test.sh XXX coct1 20345 configs/mask_rcnn_repmobile_small_fpn_1x_coco.py 'work_dirs/mask_rcnn_repmobile_large_fpn_1x_coco/epoch_12.pth' 1  

```

### **Semantic Segmentation on ADE20K**

```
cd segmentation
###Trainig

# sh tools/dist_train.sh gvembodied repseg3 configs/sem_fpn/fpn_repmobile_large_ade20k_40k.py 8 20233 work_dirs/fpn_repmobile_large_ade20k_40k/latest.pth

###Testing
# sh tools/dist_test.sh XXX repseg1 configs/sem_fpn/fpn_repmobile_large_ade20k_40k.py 1 21232 XXX/iter_40000.pth

```

## Main Results

### **Classification on ImageNet-1K**
|      name      | resolution | acc@1 | #param | FLOPs | download |
| :------------: | :--------: | :---: | :----: | :---: | :------: |
| RepMobile_S    | 224×224    | 77.2  |  5.7M  | 0.4G  | [ckpt](https://drive.google.com/file/d/1g8G_JO-E4Af9iOUG_alU-jXkKciuuMtu/view?usp=sharing) |
| RepMobile_M    | 224×224    | 78.8  |  8.2M  | 0.7G  | [ckpt](https://drive.google.com/file/d/1dHWQhtTfIsr1m06_JQdzF39rpwArCYDp/view?usp=sharing) |
| RepMobile_L    | 224×224    | 80.9  | 18.4M  | 1.4G  | [ckpt](https://drive.google.com/file/d/11N90R37HQ-aZDVvkozDAfuYURUFGUDGh/view?usp=sharing) |


### **Object Detection and Instance Segmentation on COCO**
| Model | $AP^b$ | $AP_{50}^b$ | $AP_{75}^b$ | $AP^m$ | $AP_{50}^m$ | $AP_{75}^m$ | Ckpt |
|:---------------|:----:|:---:|:--:|:--:|:--:|:--:|:--:|
| RepMobile_S | 39.8  |  61.9   | 43.5  |    37.2    |  58.8      |  40.1        |   [ckpt](https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m1_1_coco.pth)   |
| RepMobile_L | 41.6   | 63.2   | 45.3  | 38.6   | 60.5        | 41.5         |   [ckpt](https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m1_5_coco.pth)   |


### **Semantic Segmentation on ADE20K**
| Model | mIoU | Ckpt |
|:---------------|:----:|:--:|
| RepMobile_L |   42.2   |   [ckpt](https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m1_1_ade20k.pth)   |


## Acknowledgement

Classification (ImageNet) code base is partly built with [EfficientFormer](https://github.com/snap-research/EfficientFormer). 

The detection and segmentation pipeline is from [MMCV](https://github.com/open-mmlab/mmcv) ([MMDetection](https://github.com/open-mmlab/mmdetection) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)). 