# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Re-written by Jordi Gené-Mola, based on code from Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import model.faster_rcnn.vgg_models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb

class vgg16_5ch(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    self.model_path = '/work/jgene/faster_rcnn/data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    vgg = models.vgg16_5ch()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        if state_dict.items()[0][1].shape[1]!=5:
          state_dict[u'features.0.weight']=torch.cat((state_dict.items()[0][1],torch.randn(64,1,3,3),torch.randn(64,1,3,3)),1)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False


    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)      

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

