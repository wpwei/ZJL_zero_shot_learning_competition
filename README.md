# ZhejiangLab Cup 2018 - Zero Shot Learning - Baseline

[2018之江杯全球人工智能大赛 - 零样本图像目标识别](
https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.6ba733afAph1aQ&raceId=231677) 简易baseline

## 性能
|   Local   | Leaderboard |
|-----------|-------------|
|   4.7%    |     TBD     |

## 目的

* 给后来的同学一个start point
* 天池的比赛很少有Kaggle那样的分享氛围，自己在Kaggle上学了很多，希望天池也能设置鼓励分享的激励机制，让比赛热闹起来。
* **求交流求讨论求组队**（国外没人玩天池啊。。。orz。。。）

## Requirements

* python >= 3.6
* pytorch >= 0.4
* torchvision
* numpy
* pandas
* scikit-learn
* tqdm

## 用法

### 准备数据

把`DatasetA_test_20180813.zip`和`DatasetA_train_20180813.zip`解压在`input/DatasetA`下。

(这个repo已经准备好图片以外的数据，请复制train和test文件夹下的图片到对应的路径下)

### 训练模型并生成提交结果

`$ cd src; python main.py --gpu_id 0`

待提交结果保存在`output/result.txt`。

## 详细说明

### 采用模型

简单起见，本baseline采用了Google的DeViSE[1]。

首先在训练集上学习DenseNet分类器。

训练时，通过DenseNet[2]提取图像特征后，与Label的word embedding同时投影到1024维的语义空间，取内积后计算hinge rank loss训练。算法具体请参照[1]。

预测时，计算待预测图片和**全部230分类**的word embedding的语义空间投影的内积，取最大分类。

### 数据划分

|  数据集  |  赛题文件                         |
|---------|----------------------------------|
|  train  | train.txt 中除去 submit.txt 的部分 |
|   val   | submit.txt                       |
|  test   | image.txt                        | 

但这个比赛的问题是GZSL的设定（即test set包含train set的类别，不全是unseen类别， 具体请参照[3]），val set也应该包含一部分train set的数据。**欢迎讨论更靠谱的val set的分割方法。**

### 便利工具
`scr/utils.py`包含了一些方便的工具类和函数，比如数据读取，性能评价，结果保存等等，请按需取用，**同时欢迎同学们的添加自己的工具方便大家**（正坐敲碗等着pull request）。

## TODOs
* [x] 本地性能测试
* [ ] 线上性能测试

## References
[1] [DeViSE: A Deep Visual-Semantic Embedding Model](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/41473.pdf)

[2] [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

[3] [Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly](https://arxiv.org/abs/1707.00600)

    