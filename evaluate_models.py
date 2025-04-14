import torch
from vocab import Vocabulary
import evaluation_models
import torch.serialization
import argparse 

torch.serialization.add_safe_globals([argparse.Namespace])


# for coco
print('Evaluation on COCO:')
evaluation_models.evalrank("/data/coding/upload-data/data/pretrain_model/coco/model_coco_1.pth.tar", "/data/coding/upload-data/data/pretrain_model/coco/model_coco_2.pth.tar", data_path='/data/coding/upload-data/data/data/', split="testall", fold5=True)

# for flickr
print('Evaluation on Flickr30K:')
evaluation_models.evalrank("/data/coding/upload-data/data/pretrain_model/flickr/model_fliker_1.pth.tar", "/data/coding/upload-data/data/pretrain_model/flickr/model_fliker_2.pth.tar", data_path='/data/coding/upload-data/data/data/', split="test", fold5=False)