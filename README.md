# VSRN项目
原项目基于Python2.7、torch0.4.1撰写，版本过于老旧，现在租服务器很难满足这样的环境版本需求，在此对原始的VSRN项目基于Python3.10、torch2.0进行重写。

VSRN项目是在跨模态系统中一个非常经典的图文检索任务，能够很好地帮助我们更好地掌握跨模态对齐的基础技术以及复现代码的能力。

## 作业说明

作业说明：

1.阅读《Visual Semantic Reasoning for Image-Text Matching》论文并复现VSRN模型在Flickr30k数据集上的结果（评价指标采用召回率）。

2.Flickr30k数据集介绍：https://shannon.cs.illinois.edu/DenotationGraph/

论文：https://ieeexplore.ieee.org/abstract/document/9010696

代码：https://github.com/KunpengLi1994/VSRN

预处理好的数据及预训练好的模型：https://pan.baidu.com/s/16oI9sVEblGTVxf1og1JtaQ（提取码：nx1z）

3.本次作业需要提交一份报告以及代码。报告内容可参考报告模板。请将报告及代码分别命名为：姓名-学号-作业二报告.pdf、姓名-学号-作业二代码.zip，一并提交到国科大在线。

4.作业二提交截止北京时间2025年4月30日20时。

## 环境配置
```shell
conda create -n vsrn python=3.10
conda activate vsrn
pip install torch torchvision -i https://pypi.mirrors.ustc.edu.cn/simple/
# 安装pycocoevalcap和pycocotools后不再需要coco-caption和cocoapi-master文件夹
pip install pycocoevalcap pycocotools -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install tensorboard_logger -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install nltk -i https://pypi.mirrors.ustc.edu.cn/simple/
python -c "import nltk;nltk.download()"
```
