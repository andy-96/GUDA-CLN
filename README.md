# ⚠️ Code is still in reviewing process and will be uploaded soon

# GUDA-CLN

<!--![example lane detection](assets/example.gif)-->

## Motivation

Since the AVT dataset does not have any ground truth annotations and labeling the vast amount of data manually is costly and time-consuming, pre-trained supervised lane detection models are performing suboptimally as they are trained on another dataset. Due to the difference in data distribution between the AVT dataset and the dataset the model was trained on, a domain shift is introduced. This model leverages [self-supervised monocular depth estimation](https://github.com/nianticlabs/monodepth2) as an auxiliary task to overcome the domain gap and the current [SOTA-lane detection model](https://github.com/aliyun/conditional-lane-detection) as a baseline lane detector.

## Architecture
![picture of architecture](assets/cln-guda-architecture.jpg)

This work proposes a novel geometric unsupervised domain adaptation for lane detection (GUDA-Lane) to overcome the domain shift between a source dataset and target dataset. The model is composed of three networks:

- A lane detection network <img src="https://render.githubusercontent.com/render/math?math=f_L: I \rightarrow \hat{M}"> that takes an image <img src="https://render.githubusercontent.com/render/math?math=I"> as input and outputs a set of lanes <img src="https://render.githubusercontent.com/render/math?math=\hat{M}">

- A depth estimation network <img src="https://render.githubusercontent.com/render/math?math=f_D: I \rightarrow \hat{D}"> that takes an input image <img src="https://render.githubusercontent.com/render/math?math=I"> as input and outputs a estimated depth map <img src="https://render.githubusercontent.com/render/math?math=\hat{D}">

- A pose estimation network <img src="https://render.githubusercontent.com/render/math?math=f_P: {I_s, I_t} \rightarrow \hat{T}_{s\rightarrow t}"> that takes a source $I_s$ and a target image <img src="https://render.githubusercontent.com/render/math?math=I_t"> as input and outputs a transformation <img src="https://render.githubusercontent.com/render/math?math=\hat{T}_{s\rightarrow t}"> between them

To perform the domain adaptation, a feature sharing module (FSM) between the two image encoders of the lane detection and depth estimation network is introduced.

## How to run

### Setup

Create and start conda environment

```
conda env create -f environment.yml
conda activate guda-cln
```

#### DLA backbone

When starting this project, [LaneAF](https://github.com/sel118/LaneAF) was used as the lane detection baseline. However, as it is quite computationally complex, I opted for CondLaneNet. LaneAF itself does work in this project, however, it is not possible to use the domain adaptation as it would need to be implemented first. Since LaneAF uses the DLA backbone instead of a ResNet, it is also included in this repo. So, if you want to use it, run the following before using it:

```
cd networks
git clone git@github.mit.edu:chenandy/DCNv2.git
cd DCNv2
./make.sh
```

### Data Prep

This model uses [TuSimple](https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection) as the source domain, i.e. utilizing its labels, and [BDD100K](https://www.bdd100k.com/) as the target domain, i.e. only utilizing the monocular video sequences.

To prepare the datasets, copy `./data` to anywhere you want to save your data. Specific download routines have already been created, so to download and prepare the datasets, run the following:

```
# For TuSimple
cd <path-to-data>/data/tusimple
./setup_tusimple.sh
cd <path-to-guda-cln>
python datasets/utils.py --task extract-lines-txt --dataset tusimple --data_path <path-to-data>/data/tusimple

# For BDD100K
cd <path-to-data>/data/bdd100k
./setup_bdd100k.sh
cd <path-to-guda-cln>
python datasets/utils.py --task extract-lines-txt --dataset bdd100k --data_path <path-to-data>/data/bdd100k
```

To inference the model on any other dataset, the data must be stored as images.

### Training

A sample script for training the model is provided in `./scripts/sample_train.sh`. Each argument is explained there and also in `./options.py`.

The final configuration parameters of the model are set as the default values in the argument parser. As BDD100K is a very complex dataset, the depth estimation was pretrained on BDD100K. First starting with training the split `bdd100k_highway_single` for 10 epochs, then taking the pre-trained weights and training it again on `bdd100k_highway_4` and `bdd100k_highway_34` for 2 epochs, respectively. Then, the learned weights are used to continue training GUDA-CLN.

The model works only in single GPU mode. Depending on the GPU and harddrive used, the training time ranges from approx. 15h (Nvidia A100) to approx. 36h (Nvidia GTX1080).

If you want to train the model on a different target dataset, it is of benefit to pretrain the depth estimation network on the more difficult dataset for a couple of epochs and use the pretrained weights for doing the domain adapted training.

### Evaluating the model

Depending on the used dataset, there are two common ways of evaluating the model performance. For TuSimple, there is the TuSimple-way, for [CULane](https://xingangpan.github.io/projects/CULane.html), the CULane way. Both evaluation methods are implemented in this Code, however, only the CULane way can be used for all datasets whereas the TuSimple way only works for TuSimple.

To evaluate the model, the evaluation script for the CULane evaluation method needs to be build. To do that, run the following:

```
cd evaluation/CULane
./build.sh
```

For evaluating a model, take a look at `./scripts/sample_eval.sh`.


### Inference

Please follow the jupyter notebook for training, evaluation, and inference information: `./sample.ipynb`

## Results

GUDA-CLN was compared against the baseline CondLaneNet implementation. 

**Evaluation on TuSimple**

The first table shows the evaluation results on the source domain indicating how well the lane detection task itself performs. Both GUDA-CLN and the baseline model perform similarly well, however, the performance on the TuSimple is already saturated.

| Method | Source | Target | FN &#8595; | FP &#8595; | Acc &#8593; |
| --- | --- | --- | --- | --- | --- |
| CondLaneNet | TuSimple | - | 6.3% | 4.1% | 94.3% |
| GUDA-CLN | TuSimple | BDD100K | **5.4%** | **3.9%** | **94.7%** |

**Evaluation on BDD100K**

This table indicates a slight performance increase on the target dataset of GUDA-CLN in comparison to CondLaneNet which suggests the efficacy of the domain adaptation.

| Method | Source | Target | Prec &#8593; | Rec &#8593; | Acc &#8593; |
| --- | --- | --- | --- | --- | --- |
| CondLaneNet | TuSimple | - | 48.1% | **30.8%** | 37.5% |
| GUDA-CLN | TuSimple | BDD100K | **56.6%** | 29.8% | **39.0%** |
Evaluated on BDD100K

**Evaluation on BDD100K (only ego lanes)**

For the lane variability analysis, only the ego-lanes, i.e. the nearest lanes left and right of the centerline of the image, are necessary to compute the lateral position of the vehicle within a lane. Thus, as a second evaluation, only the ego-lanes are taken into consideration further increase the performance boost of GUDA-CLN over CondLaneNet.

| Method | Source | Target | Prec &#8593; | Rec &#8593; | Acc &#8593; |
| --- | --- | --- | --- | --- | --- |
| CondLaneNet | TuSimple | - | 41.8% | **39.5%** | 40.6% |
| GUDA-CLN | TuSimple | BDD100K | **50.2%** | 39.4% | **44.2%** |

**Evaluation on AVT**

The first model was initially used as the lane detection model and it is clearly apparent, that GUDA-CLN outperforms this model by a large margin.

| Method | Source | Target | Prec &#8593; | Rec &#8593; | Acc &#8593; |
| --- | --- | --- | --- | --- | --- |
| SimCycleGAN + ERFNet | TuSimple | - | 33.4% | 33.6% | 33.5% |
| CondLaneNet | TuSimple | BDD100K | <ins>63.0%</ins> | <ins>60.3%</ins> | <ins>61.6%</ins> |
| GUDA-CLN | TuSimple | BDD100K | **74.2%** | **62.6%** | **67.9%** |

## Pretrained models

In the following, the pretrained lane detection models are saved. The pretrained BDD100K on Monodepth2 can be used as pretrained weights, if you want to retrain GUDA-CLN on TuSimple as a source domain and BDD100K as a target domain. 

| Training Modality | Links |
| --- | --- |
| Pre-trained BDD100K on Monodepth2                      | [Download](https://www.dropbox.com/s/o4bqk6zvn8qf0z1/md-bdd.tar.gz?dl=0) |
| GUDA-CLN from TuSimple to BDD100K | [Download](https://www.dropbox.com/s/hr0cvxg6ojkww0h/da-cln-tus-bdd.tar.gz?dl=0) |
| CondLaneNet trained on BDD100K                         | [Download](https://www.dropbox.com/s/baklo5aqyxo6dcp/cln-bdd.tar.gz?dl=0) |
