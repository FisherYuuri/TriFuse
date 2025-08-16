# TriFuse: Revisiting Visible-Thermal-Depth Salient Object Detection through an Unbiased Quality Lens

This repo is an official implementation of the *TriFuse*.
**TriFuse: Revisiting Visible-Thermal-Depth Salient Object Detection through an Unbiased Quality Lens**

## Prerequisites
- The **VDT-RW** dataset is available at:https://pan.baidu.com/s/1rnOPWGQjthZdN3Qmqq7pUw?pwd=mvua.
- The VDT-2048 dataset is available at:https://pan.baidu.com/s/1JyFBtjlJGf4GE2zeciN1wQ?pwd=bipy.
- The pretrained weights for the backbone networks can be downloaded at:
-  [SwinTransformer](https://pan.baidu.com/s/1lRKC_caVWzVuJwvVfsCWYg?pwd=3hj7).
-  [Mobilenetv3](https://pan.baidu.com/s/1PDAgND6AxwZHUFlkx2KOTg?pwd=a4c8).
-  [VGG16](https://pan.baidu.com/s/1QA7IPUp2su2a9QXYiB4GBg?pwd=46ts).

## Usage

### 1. Clone the repository
## Setup
```
conda create -n TriFuse python==3.10
conda activate TriFuse
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```
### 2. Training
After adjusting the dataset path, you can train the model by using the following command:
```
cd Train
python TrainTriFuse.py
python TrainTriFuseVD.py
python TrainTriFuseVT.py
```

### 3. Testing
After adjusting the dataset path, you can test the model by using the following command:
```
cd Test
python Test.py
python TestVD.py
python TestVT.py
```

### 4. Evaluation
The following table provides links to the pre-trained weights and saliency map results of TriFuse on various datasets:
| Dataset   | Backbone         | Saliency Maps                                             | Model Weights                                             |
|-----------|------------------|------------------------------------------------------------|-----------------------------------------------------------|
| VDT-2048  | Swin Transformer | [Download](https://pan.baidu.com/s/1BPfIGuORWIFPvaljxTNxNA?pwd=bydf)| [Download](https://pan.baidu.com/s/1FlwS9pdcuVLw13ispbhlWA?pwd=a2nr)|
| VDT-RW    | Swin Transformer | [Download](https://pan.baidu.com/s/1STXaAxphKCH8clVbfyt8GQ?pwd=9jee)| [Download](https://pan.baidu.com/s/1n9zDe5OityRFuQhMke4hIg?pwd=setk)|
| VDT-2048  | Mobilenetv3 | [Download](https://pan.baidu.com/s/1BPfIGuORWIFPvaljxTNxNA?pwd=bydf)| [Download](https://pan.baidu.com/s/1JLRsZVvmi4lIlT4cEfbKdA?pwd=ax8x)|
| VDT-2048  | ResNet50 | [Download](https://pan.baidu.com/s/1BPfIGuORWIFPvaljxTNxNA?pwd=bydf)| [Download](https://pan.baidu.com/s/12vXLlNKhsgMa1DiLbHmtoA?pwd=e5ur)|
| VDT-2048  | VGG16 | [Download](https://pan.baidu.com/s/1BPfIGuORWIFPvaljxTNxNA?pwd=bydf)| [Download](https://pan.baidu.com/s/1ca69iRIYO5Y0cxvWHNqKZA?pwd=hx34)|
| VT-1000   | Swin Transformer | [Download](https://pan.baidu.com/s/1rPHZJ1ijpdE6KBLj5jbqrw?pwd=jus4)| [Download](https://pan.baidu.com/s/1yYxmMceL_-WPBJXHGcPBYA?pwd=i39u)|
| STERE     | Swin Transformer | [Download](https://pan.baidu.com/s/19zmjO9ttny450DI3VEzX7Q?pwd=qkuv)| [Download](https://pan.baidu.com/s/1GLqzzNCZvQgVJDr2Jymw2w?pwd=mbir)|


- We adopt the [evaluation toolbox](https://github.com/DengPingFan/SINet) provided by the SINet repository to compute quantitative metrics. 
- We provide the saliency maps for the challenging sub-datasets of TriFuse at the following link:[Download](https://pan.baidu.com/s/1uGtB9cu89eTaHmRZbHBuGw?pwd=bra7)
- We provide the saliency maps of competing methods and challenging sub-datasets on VDT2048, which can be accessed via the following link:[Download](https://pan.baidu.com/s/19-waBKdIR0fFYrNQS3J86g?pwd=usqg)
- We provide the saliency maps of competing methods on VDTRW, which can be accessed via the following link:[Download](https://pan.baidu.com/s/1STXaAxphKCH8clVbfyt8GQ?pwd=9jee)

## To do
- [ ] Upload all files to Google Drive.

## Acknowledgements
We would like to thank the authors of the following projects for their excellent work and contributions to the community:
- https://github.com/Lx-Bao/IFENet  
- https://github.com/DengPingFan/SINet  
- https://github.com/zyrant/LSNet  
- https://github.com/CSer-Tang-hao/ConTriNet_RGBT-SOD  
- https://github.com/ENSTA-U2IS-AI/infraParis
