# Learning Sensor-Specific Spatial-Spectral Features of Hyperspectral Images via Convolutional Neural Networks

By [Shaohui Mei](http://teacher.nwpu.edu.cn/en/meishaohui.html), Jingyu Ji, [Junhui Hou](http://sites.google.com/site/junhuihoushomepage/), [Xu Li](http://teacher.nwpu.edu.cn/en/lixu.html), [Qian Du](http://my.ece.msstate.edu/faculty/du/).

---
### Introduction


The C-CNN is an unified framework for hyperspectral image classification with a single network. You can use the code to train/evaluate a network for hsi classification. For more details, please refer to our [paper](http://ieeexplore.ieee.org/document/7919223/) and our [slide](http://pan.baidu.com/s/1qXF2XcC).

<p align="center">
<img src="http://images2015.cnblogs.com/blog/706487/201706/706487-20170604171211430-1803403755.png" 
alt="Proposed Framework" width="600px"><br /><a> Proposed Framework</a>
</p>

The Feature of AVIRIS sensor are as follow.
<p align="center">
<img src=http://images2015.cnblogs.com/blog/706487/201706/706487-20170604184141368-132139908.jpg
alt="the Feature of AVIRIS sensor" width="600px"><br /><a> the Feature of AVIRIS sensor</a>
</p>

Some classification results (overall accuracy) are listed below:

| Datasets | SVM-RBF | Proposed C-CNN (1x1) | 3x3 mean | 5x5 mean | 3x3 mean + std | 5x5 mean + std |
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Indian | 84.11 | 85.75 | 94.62 | 96.38 | 94.71 | **96.76** |
| Salina | 81.55 | 92.19 | 96.40 | 97.73 | 95.85 | **97.42** |

---
### Citing our work

Please cite our work in your publications if it helps your research:
```latex
@ARTICLE{7919223,
    author={S. Mei and J. Ji and J. Hou and X. Li and Q. Du},
    journal={IEEE Transactions on Geoscience and Remote Sensing},
    title={Learning Sensor-Specific Spatial-Spectral Features of Hyperspectral Images via Convolutional Neural Networks},
    year={2017},
    volume={PP},
    number={99},
    pages={1-14},
    keywords={Feature extraction;Hyperspectral imaging;Image sensors;Machine learning;Principal component analysis;Sensors;Classification;convolutional neural network (CNN);feature learning;hyperspectral;spatial-spectral.},
    doi={10.1109/TGRS.2017.2693346},
    ISSN={0196-2892},
    month={},
}
```

---
### Installation
##### Install Caffe 
1. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
```Shell
git clone https://github.com/BVLC/caffe.git
# Modify Makefile.config according to your Caffe installation.
cp Makefile.config.example Makefile.config
make -j8
# Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
make py
make test -j8
# (Optional)
make runtest -j8
```
- Clone this repository. 
 - Note: We currently only support Python 2.7 

---
### Preparation

##### Modify Caffe root

if necessary, modify caffe root in the code:

```Shell
gedit ./src/net/find_caffe.py
```

##### Download datasets
The Indian Pines and Salina Valley will be download by the following stript:
```bash
bash ./src/get_HSI_data.sh
```

##### Preprocess
In our paper, amount of spatial information about hyperspectral image was explored. In our code, 1x1, 3x3 mean, 5x5 mean, 3x3 mean+std, 5x5 mean+std was listed to compare.

```Shell
cd ./src
python data_analysis/pre_process.py
```

---
### Traineval 

- To train CNN using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.
```Shell
cd ./src
python train_cnn.py
```
- Training Parameter Options:
```python
parser = argparse.ArgumentParser(description="train bn net", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--spatial_info', type=str, default='5x5_mean_std', help="'1x1_mean' | '3x3_mean' | '3x3_mean_std' | '5x5_mean'| '5x5_mean_std'")
parser.add_argument('--gpu', type=int, default=0, help='the number of gpu id, only one number is required')
parser.add_argument('--dst_dir', type=str, default='bn_net_200', help='the destination dir for the experiments')
parser.add_argument('--data_set', type=str, default='indian_pines', help='indian_pines, salina')
parser.add_argument('--max_iter', type=int, default=1000000, help='how many iters')
parser.add_argument('--train_nums', type=float, default=200, help='how many samples or how much percents for training, 200 or 0.1')
args = parser.parse_args()
```
---
### Feature extraction

- Use the following script to extract the supervised deep feature or sensor-specific feature of datasets acquired by AVIRIS sensor.
```Shell
cd ./src
python feature_extract.py
```
