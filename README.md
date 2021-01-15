# EdgeFace_Pytorch
Official EdgeFace Pytorch code
<p align="center"><img src="data/result.png" width=\linewidth\></p>

## WiderFace Val Performance
| Model | easy | medium | hard |
|:-|:-:|:-:|:-:|
| EdgeFace | 90.5 % | 89.8% | 83.8% |


## Installation
##### Clone and install
1. git clone https://github.com/JardinRyu/EdgeFace_Pytorch.git

2. Pytorch version 1.1.0+ and torchvision 0.3.0+ are needed.

3. Codes are based on Python 3

##### Data
1. Download the [WIDERFACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) dataset.

2. Download annotations (face bounding boxes & five facial landmarks) from [baidu cloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA) or [dropbox](https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0)

3. Organise the dataset directory as follows:

```Shell
  ./data/widerface/
    train/
      images/
      label.txt
    val/
      images/
      wider_val.txt
```
ps: wider_val.txt only include val file names but not label information.

##### Data1
We also provide the organized dataset we used as in the above directory structure.

Link: from [google cloud](https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS) or [baidu cloud](https://pan.baidu.com/s/1jIp9t30oYivrAvrgUgIoLQ) Password: ruck

## Training

```Shell
python train.py
```

## Evaluation
### Evaluation widerface val
1. Generate txt file
```Shell
python test_widerface.py --trained_model [weight_file] (default='./weights/EdgeFaceNet_Final.pth')
```
2. Evaluate txt results.
```Shell
cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```

## References
- [EXTD_Pytorch](https://github.com/clovaai/EXTD_Pytorch)
- [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)

