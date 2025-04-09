# VSRN项目
[VSRN原项目](https://github.com/KunpengLi1994/VSRN/tree/master)基于Python2.7、torch0.4.1撰写，版本过于老旧，现在租服务器很难满足这样的环境版本需求，在此对原始的VSRN项目基于Python3.10、torch2.0进行重写。

VSRN项目是在跨模态系统中一个非常经典的图文检索任务，能够很好地帮助我们更好地掌握跨模态对齐的基础技术以及复现代码的能力。

## 要求

阅读《Visual Semantic Reasoning for Image-Text Matching》论文并复现VSRN模型在Flickr30k数据集上的结果（评价指标采用召回率）。


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
