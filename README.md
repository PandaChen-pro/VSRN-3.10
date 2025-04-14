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
# 安装pycocotools后不再需要cocoapi-master文件夹
apt-get update
apt install openjdk-11-jre
pip install  pycocotools -i https://pypi.mirrors.ustc.edu.cn/simple/

pip install tensorboard -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install tensorboard_logger -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install nltk -i https://pypi.mirrors.ustc.edu.cn/simple/
```
## 下载punkt包
```shell
# 安装shell crash
export url='https://fastly.jsdelivr.net/gh/juewuy/ShellCrash@master' && wget -q --no-check-certificate -O /tmp/install.sh $url/install.sh  && bash /tmp/install.sh && source /etc/profile &> /dev/null
# 翻墙机场:https://www3rd.ga-sub.lat/api/v1/client/subscribe?token=ad0c48690287bf56df1787929bc06ecc
# 元机场:https://185.213.174.24/search?token=aa7b304f1804142441a2410a51876b2b
python ./utils/download_nltk.py
```

## 执行evaluate
修改`/data/coding/VSRN-3.10/evaluate_models.py`，修改模型和数据所在路径。
```python
# MsCoCo
evaluation_models.evalrank("/data/coding/upload-data/data/pretrain_model/coco/model_coco_1.pth.tar", "/data/coding/upload-data/data/pretrain_model/coco/model_coco_2.pth.tar", data_path='/data/coding/upload-data/data/data/', split="testall", fold5=True)
# Fliker
evaluation_models.evalrank("/data/coding/upload-data/data/pretrain_model/flickr/model_fliker_1.pth.tar", "/data/coding/upload-data/data/pretrain_model/flickr/model_fliker_2.pth.tar", data_path='/data/coding/upload-data/data/data/', split="test", fold5=False)
```

