
# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Written by Jordi Gené-Mola, based on code from jwyang
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import pickle

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.vgg16_4ch import vgg16_4ch
from model.faster_rcnn.vgg16_5ch import vgg16_5ch
from model.faster_rcnn.resnet import resnet

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=100, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="/work/jgene/faster_rcnn/data/kinect_fruits_models",
                      nargs=argparse.REMAINDER)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfboard', dest='use_tfboard',
                      help='whether use tensorflow tensorboard',
                      default=False, type=bool)

# Color space
  parser.add_argument('--RGB', dest='RGB',
                      help='Evaluation RGB images',
                      action='store_true')
  parser.add_argument('--NIR', dest='NIR',
                      help='Evaluation NIR images',
                      action='store_true')
  parser.add_argument('--DEPTH', dest='DEPTH',
                      help='Evaluation DEPTH images',
                      action='store_true')

  parser.add_argument('--anchor', action='append')

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.use_tfboard:
    from model.utils.logger import Logger
    # Set the logger
    logger = Logger('./logs')

  if args.dataset == "kinect_fruits":
      args.imdb_name = "train"
      args.imdbval_name = "val"
      args.set_cfgs = ['ANCHOR_SCALES', list(np.array(args.anchor).astype(np.int)), 'ANCHOR_RATIOS', '[0.5,1,2]',
                       'MAX_NUM_GT_BOXES', '100']
  elif args.dataset == "kinect_fruits_k":
      args.imdb_name = "train_k"
      args.imdbval_name = "val_k"
      args.set_cfgs = ['ANCHOR_SCALES', list(np.array(args.anchor).astype(np.int)), 'ANCHOR_RATIOS', '[0.5,1,2]',
                       'MAX_NUM_GT_BOXES', '100']


  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)


  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda

  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)

  cfg.TRAIN.USE_FLIPPED = False
  imdbval, roidbval, ratio_listval, ratio_indexval = combined_roidb(args.imdbval_name)  

  print('{:d} roidb training entries'.format(len(roidb)))
  print('{:d} roidb validation entries'.format(len(roidbval)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  
  
  dataset = {}
  dataset = {'train': roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, args.RGB, args.NIR, args.DEPTH, training=True),
              'val': roibatchLoader(roidbval, ratio_listval, ratio_indexval, args.batch_size, \
                           imdbval.num_classes, args.RGB, args.NIR, args.DEPTH, training=True, normalize = False)}
  set_size = {}
  '''set_size = {'train': len(roidb),
            'val': len(roidbval)}'''
  set_size = {phase: len(dataset[phase]) for phase in ['train','val']}
  train_sampler_batch = sampler(set_size['train'], args.batch_size)
  dataloader = {}
  dataloader={'train': torch.utils.data.DataLoader(dataset['train'], batch_size=args.batch_size,
                             sampler=train_sampler_batch, num_workers=args.num_workers),
              'val': torch.utils.data.DataLoader(dataset['val'], batch_size=args.batch_size,
                                shuffle=False, num_workers=0,
                                pin_memory=True)}


  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  im_data_val = torch.FloatTensor(1)
  im_info_val = torch.FloatTensor(1)
  num_boxes_val = torch.LongTensor(1)
  gt_boxes_val = torch.FloatTensor(1)


  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

    im_data_val = im_data_val.cuda()
    im_info_val = im_info_val.cuda()
    num_boxes_val = num_boxes_val.cuda()
    gt_boxes_val = gt_boxes_val.cuda()


  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  im_data_val = Variable(im_data_val)
  im_info_val = Variable(im_info_val)
  num_boxes_val = Variable(num_boxes_val)
  gt_boxes_val = Variable(gt_boxes_val)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'vgg16_4ch':
    fasterRCNN = vgg16_4ch(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'vgg16_5ch':
    fasterRCNN = vgg16_5ch(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  if args.cuda:
    fasterRCNN.cuda()

  iters_per_epoch = {phase: int(set_size[phase] / args.batch_size) for phase in ['train','val']}
  print(iters_per_epoch)
  print(iters_per_epoch['train'])
  print(iters_per_epoch['val'])
  loss_train = torch.zeros(args.max_epochs)
  loss_val = torch.zeros(args.max_epochs)
  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode
    fasterRCNN.train()
    loss_temp = 0
    start = time.time()

    for phase in ['train','val']:


      if epoch % (args.lr_decay_step + 1) == 0 and phase == 'train':
          adjust_learning_rate(optimizer, args.lr_decay_gamma)
          lr *= args.lr_decay_gamma

      loss_save = 0

      for step, data in enumerate(dataloader[phase]):
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])
        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)


        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
             + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        loss_temp += loss.data[0]
        loss_save += loss.data[0]

        if phase == 'train':
          # backward
          optimizer.zero_grad()
          loss.backward()
          if args.net == "vgg16":
              clip_gradient(fasterRCNN, 10.00)
          optimizer.step()
        if phase == 'train':
          aux = step % 100
        elif phase == 'val':
          aux = step % 20
        if aux == 0:
          end = time.time()
          if step > 0:
            if phase == 'train':
              loss_temp /= 100
            elif phase == 'val':
              loss_temp /= 20

          if args.mGPUs:
            loss_rpn_cls = rpn_loss_cls.mean().data[0]
            loss_rpn_box = rpn_loss_box.mean().data[0]
            loss_rcnn_cls = RCNN_loss_cls.mean().data[0]
            loss_rcnn_box = RCNN_loss_bbox.mean().data[0]
            fg_cnt = torch.sum(rois_label.data.ne(0))
            bg_cnt = rois_label.data.numel() - fg_cnt
          else:
            loss_rpn_cls = rpn_loss_cls.data[0]
            loss_rpn_box = rpn_loss_box.data[0]
            loss_rcnn_cls = RCNN_loss_cls.data[0]
            loss_rcnn_box = RCNN_loss_bbox.data[0]
            fg_cnt = torch.sum(rois_label.data.ne(0))
            bg_cnt = rois_label.data.numel() - fg_cnt
          
          print("[session %d][epoch %2d][iter %4d/%4d] loss%s: %.4f, lr: %.2e" \
                                  % (args.session, epoch, step, iters_per_epoch[phase], phase, loss_temp, lr))
          print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
          print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                        % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
          if args.use_tfboard:
            info = {
              'loss': loss_temp,
              'loss_rpn_cls': loss_rpn_cls,
              'loss_rpn_box': loss_rpn_box,
              'loss_rcnn_cls': loss_rcnn_cls,
              'loss_rcnn_box': loss_rcnn_box
            }
            for tag, value in info.items():
              logger.scalar_summary(tag, value, step)

          loss_temp = 0
               
          start = time.time()
      if phase == 'train':
        loss_train[epoch-1] = loss_save/iters_per_epoch[phase]
      elif phase == 'val':
        loss_val[epoch-1] = loss_save/iters_per_epoch[phase]



      if phase == 'train':
        if args.mGPUs:
          save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
          save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
          }, save_name)
        else:
          save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
          save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
          }, save_name)
        print('save model: {}'.format(save_name))

    end = time.time()
    print(end - start)

  
  pickle.dump(loss_val,open('output_train_nou/val_loss_s'+str(args.session)+'.pkl','wb'))
  mat = np.matrix(loss_val)
  with open('output_train_nou/val_loss_s'+str(args.session)+'.txt','wb') as f:
      for line in mat:
          np.savetxt(f, line, fmt='%.4f')
          
  pickle.dump(loss_train,open('output_train_nou/train_loss_s'+str(args.session)+'.pkl','wb'))
  mat = np.matrix(loss_train)
  with open('output_train_nou/train_loss_s'+str(args.session)+'.txt','wb') as f:
      for line in mat:
          np.savetxt(f, line, fmt='%.4f')
