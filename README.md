# TriFuse: Revisiting Visible-Thermal-Depth Salient Object Detection through an Unbiased Quality Lens

This repo is an official implementation of the *TriFuse*.
**TriFuse: Revisiting Visible-Thermal-Depth Salient Object Detection through an Unbiased Quality Lens**

## Prerequisites
- The **VDT-RW** dataset is available at: [BaiduDisk](https://pan.baidu.com/s/1rnOPWGQjthZdN3Qmqq7pUw?pwd=mvua)\|[GoogleDrive](https://drive.google.com/file/d/10xCSb3ELmD-oDtp8gWsArCG4FsYHvA-M/view?usp=drive_link).
- The VDT-2048 dataset is available at: [BaiduDisk](https://pan.baidu.com/s/1JyFBtjlJGf4GE2zeciN1wQ?pwd=bipy).
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
| VDT-2048  | Swin Transformer | [BaiduDisk](https://pan.baidu.com/s/1BPfIGuORWIFPvaljxTNxNA?pwd=bydf)\|[GoogleDrive](https://drive.google.com/file/d/1bVerQXkg87jJe7wl80FYEsWAedUKinwt/view?usp=drive_link)| [BaiduDisk](https://pan.baidu.com/s/1FlwS9pdcuVLw13ispbhlWA?pwd=a2nr)\|[GoogleDrive](https://drive.google.com/file/d/1vtWaBGSaWhzlrTmi3SNrVIcgc6vily4A/view?usp=drive_link)|
| VDT-RW    | Swin Transformer | [BaiduDisk](https://pan.baidu.com/s/1STXaAxphKCH8clVbfyt8GQ?pwd=9jee)\|[GoogleDrive](https://drive.google.com/file/d/1bUPEJO4Vuzk0H1-OkaY1gJOj8EPS69rQ/view?usp=drive_link)| [BaiduDisk](https://pan.baidu.com/s/1n9zDe5OityRFuQhMke4hIg?pwd=setk)\|[GoogleDrive](https://drive.google.com/file/d/1CGledkw5K7Efvoo-wzOcHBGEs1wHuNRj/view?usp=drive_link)|
| VDT-2048  | Mobilenetv3 | [BaiduDisk](https://pan.baidu.com/s/1BPfIGuORWIFPvaljxTNxNA?pwd=bydf)\|[GoogleDrive](https://drive.google.com/file/d/1bVerQXkg87jJe7wl80FYEsWAedUKinwt/view?usp=drive_link)| [BaiduDisk](https://pan.baidu.com/s/1JLRsZVvmi4lIlT4cEfbKdA?pwd=ax8x)\|[GoogleDrive](https://drive.google.com/file/d/1JfORbOqOdWLBHPsAkc5vwf_gy4VbVMyM/view?usp=drive_link)|
| VDT-2048  | ResNet50 | [BaiduDisk](https://pan.baidu.com/s/1BPfIGuORWIFPvaljxTNxNA?pwd=bydf)\|[GoogleDrive](https://drive.google.com/file/d/1bVerQXkg87jJe7wl80FYEsWAedUKinwt/view?usp=drive_link)| [BaiduDisk](https://pan.baidu.com/s/12vXLlNKhsgMa1DiLbHmtoA?pwd=e5ur)\|[GoogleDrive](https://drive.google.com/file/d/1vjBQV5r35nJY7MqvdFGrgirXnrCkUTfm/view?usp=drive_link)|
| VDT-2048  | VGG16 | [BaiduDisk](https://pan.baidu.com/s/1BPfIGuORWIFPvaljxTNxNA?pwd=bydf)\|[GoogleDrive](https://drive.google.com/file/d/1bVerQXkg87jJe7wl80FYEsWAedUKinwt/view?usp=drive_link)| [BaiduDisk](https://pan.baidu.com/s/1ca69iRIYO5Y0cxvWHNqKZA?pwd=hx34)\|[GoogleDrive](https://drive.google.com/file/d/1sIhAndQNAKIw9OvNRXbwaKN-d-z5dR4d/view?usp=drive_link)|
| VT-1000   | Swin Transformer | [BaiduDisk](https://pan.baidu.com/s/1rPHZJ1ijpdE6KBLj5jbqrw?pwd=jus4)|[GoogleDrive](https://drive.google.com/file/d/1ZeDDXnpl9mzS69gBPf0RZIoY2q3ky5dk/view?usp=drive_link)| [BaiduDisk](https://pan.baidu.com/s/1yYxmMceL_-WPBJXHGcPBYA?pwd=i39u)\|[GoogleDrive](https://drive.google.com/file/d/1u3yWhK_CsaMK48xYt2WKcTJ04Uct4TFk/view?usp=drive_link)|
| STERE     | Swin Transformer | [BaiduDisk](https://pan.baidu.com/s/19zmjO9ttny450DI3VEzX7Q?pwd=qkuv)\|[GoogleDrive](https://drive.google.com/file/d/10tIbUwYijoXYIkoevSG74HZjBcgFDftK/view?usp=drive_link)| [BaiduDisk](https://pan.baidu.com/s/1GLqzzNCZvQgVJDr2Jymw2w?pwd=mbir)\|[GoogleDrive](https://drive.google.com/file/d/1b-3IOFmjlAgu4-Hkhy-griaa9vA9np5c/view?usp=drive_link)|


- We adopt the [evaluation toolbox](https://github.com/DengPingFan/SINet) provided by the SINet repository to compute quantitative metrics. 
- We provide the saliency maps for the challenging sub-datasets of TriFuse at the following link:[Download](https://pan.baidu.com/s/1uGtB9cu89eTaHmRZbHBuGw?pwd=bra7)
- We provide the saliency maps of competing methods and challenging sub-datasets on VDT2048, which can be accessed via the following link:[Download](https://pan.baidu.com/s/19-waBKdIR0fFYrNQS3J86g?pwd=usqg)
- We provide the saliency maps of competing methods on VDTRW, which can be accessed via the following link:[BaiduDisk](https://pan.baidu.com/s/1STXaAxphKCH8clVbfyt8GQ?pwd=9jee)\|[GoogleDrive](https://drive.google.com/file/d/1bUPEJO4Vuzk0H1-OkaY1gJOj8EPS69rQ/view?usp=drive_link)

## Acknowledgements
We would like to thank the authors of the following projects for their excellent work and contributions to the community:
- https://github.com/Lx-Bao/IFENet  
- https://github.com/DengPingFan/SINet  
- https://github.com/zyrant/LSNet  
- https://github.com/CSer-Tang-hao/ConTriNet_RGBT-SOD  
- https://github.com/ENSTA-U2IS-AI/infraParis
