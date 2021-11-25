import argparse
#from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
from torchvision import transforms
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
import socket
from datetime import datetime
import os
import torch
from tensorboardX import SummaryWriter
import tqdm
import torch.nn as nn
from torch.nn import functional as F
import math
import torch.backends.cudnn as cudnn
from torch.nn import init
from models.rle2mask import files_mask2rle
from models.utils.segmentation import MultiHeadAttention
from models.Base import DeepLabV3 as base
from models.transformers.Cell_DETR import cell_detr_128
# from dice_loss import SoftDiceLoss, get_soft_label, val_dice_fetus, val_dice_isic
# from dice_loss import Intersection_over_Union_fetus, Intersection_over_Union_isic
# 1212112
###########################################################
'''
2.默认配置
'''
###########################################################
ROOT_path = os.path.abspath('../../../')


class DefaultConfig(object):
    num_epochs = 475
    epoch_start_i = 0
    checkpoint_step = 5
    validation_step = 1
    crop_height = 256
    crop_width = 448
    batch_size = 2
    # 训练集所在位置，根据自身训练集位置进行修改
    data = r'dataset/OCT/trainingset/'

    # log_dirs = os.path.join(ROOT_path, 'Log/OCT')
    log_dirs = 'Log/OCT'

    lr = 0.04
    lr_mode = 'poly'
    net_work = 'BaseNet'
    # net_work= 'MSSeg'  #net_work= 'UNet'

    momentum = 0.9  #
    weight_decay = 1e-4  #

    mode = 'train'
    num_classes = 2

    k_fold = 4
    test_fold = 4
    num_workers = 0

    cuda = '2'
    use_gpu = True
    pretrained_model_path = os.path.join(ROOT_path, 'pretrained', 'resnet34-333f7ec4.pth')
    save_model_path = './check_aCPF_BAT'
###########################################################
'''
3.diceloss
'''
###########################################################
from torch.autograd import Variable


class DiceLoss(nn.Module):
    def __init__(self, smooth=0.01):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        input = torch.sigmoid(input)
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        intersect = (input * target).sum()
        union = torch.sum(input) + torch.sum(target)
        Dice = (2 * intersect + self.smooth) / (union + self.smooth)
        dice_loss = 1 - Dice
        return dice_loss


class Multi_DiceLoss(nn.Module):
    def __init__(self, class_num=5, smooth=0.001):
        super(Multi_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self, input, target):
        input = torch.exp(input)
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(0, self.class_num):
            input_i = input[:, i, :, :]
            target_i = (target == i).float()
            intersect = (input_i * target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += dice
        dice_loss = 1 - Dice / (self.class_num)
        return dice_loss


class EL_DiceLoss(nn.Module):
    def __init__(self, class_num=4, smooth=1, gamma=0.5):
        super(EL_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num
        self.gamma = gamma

    def forward(self, input, target):
        input = torch.exp(input)
        self.smooth = 0.
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(1, self.class_num):
            input_i = input[:, i, :, :]
            target_i = (target == i).float()
            intersect = (input_i * target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = Variable(torch.Tensor([1]).float()).cuda()
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += (-torch.log(dice)) ** self.gamma
        dice_loss = Dice / (self.class_num - 1)
        return dice_loss
###########################################################
'''
4.图像增强
'''
###########################################################
def augmentation():
    # augment images with spatial transformation: Flip, Affine, Rotation, etc...
    # see https://github.com/aleju/imgaug for more details
    pass


def augmentation_pixel():
    # augment images with pixel intensity transformation: GaussianBlur, Multiply, etc...
    pass


class Data(torch.utils.data.Dataset):
    Unlabelled = [0, 0, 0]
    sick = [255, 255, 255]
    COLOR_DICT = np.array([Unlabelled, sick])

    def __init__(self, dataset_path, scale=(320, 320), mode='train'):
        super().__init__()
        self.mode = mode
        self.img_path = dataset_path + '/img'
        self.mask_path = dataset_path + '/mask'
        self.image_lists, self.label_lists = self.read_list(self.img_path)
        self.resize = scale
        self.flip = iaa.SomeOf((2, 5), [
            iaa.PiecewiseAffine(scale=(0, 0.1), nb_rows=4, nb_cols=4, cval=0),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.1),
            iaa.Affine(rotate=(-20, 20),
                       scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
            iaa.OneOf([
                iaa.GaussianBlur((0, 1.0)),  # blur images with a sigma between 0 and 3.0
                iaa.AverageBlur(k=(3, 5)),  # blur image using local means with kernel sizes between 2 and 7
                iaa.MedianBlur(k=(3, 5)),  # blur image using local medians with kernel sizes between 2 and 7
            ]),
            iaa.contrast.LinearContrast((0.5, 1.5))],
                               random_order=True)

        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # load image and crop
        img = Image.open(self.image_lists[index]).convert('RGB')
        img = img.resize(self.resize)
        img = np.array(img)
        labels = self.label_lists[index]
        # load label
        if self.mode != 'test':
            label_ori = Image.open(self.label_lists[index]).convert('RGB')
            label_ori = label_ori.resize(self.resize)
            label_ori = np.array(label_ori)
            label = np.ones(shape=(label_ori.shape[0], label_ori.shape[1]), dtype=np.uint8)

            # convert RGB  to one hot

            for i in range(len(self.COLOR_DICT)):
                equality = np.equal(label_ori, self.COLOR_DICT[i]) # 先找无标签，找到标签的索引返回ture
                class_map = np.all(equality, axis=-1)
                label[class_map] = i

            # augment image and label
            if self.mode == 'train':
                seq_det = self.flip.to_deterministic()  # 固定变换
                segmap = ia.SegmentationMapsOnImage(label, shape=label.shape)
                img = seq_det.augment_image(img)
                label = seq_det.augment_segmentation_maps([segmap])[0].get_arr().astype(np.uint8)

            label_img = torch.from_numpy(label.copy()).float()
            if self.mode == 'val':
                img_num = len(os.listdir(os.path.dirname(labels)))
                labels = label_img, img_num
            else:
                labels = label_img
        imgs = img.transpose(2, 0, 1) / 255.0
        img = torch.from_numpy(imgs.copy()).float()  # self.to_tensor(img.copy()).float()
        return img, labels

    def __len__(self):
        return len(self.image_lists)

    def read_list(self, image_path):
        fold = os.listdir(image_path)
        # fold = sorted(os.listdir(image_path), key=lambda x: int(x[-2:]))
        # print(fold)

        img_list = []
        label_list = []
        if self.mode == 'train':
            for item in fold:
                name = item.split('.')[0]
                img_list.append(os.path.join(image_path, item))
                label_list.append(os.path.join(image_path.replace('img', 'mask'), '{}.png'.format(name)))


        elif self.mode == 'val':
            for item in fold:
                name = item.split('.')[0]
                img_list.append(os.path.join(image_path, item))
                label_list.append(os.path.join(image_path.replace('img', 'mask'), '{}.png'.format(name)))


        elif self.mode == 'test':
            for item in fold:
                name = item.split('.')[0]
                img_list.append(os.path.join(image_path, item))
                label_list.append(os.path.join(image_path.replace('img', 'mask'), '{}.png'.format(name)))

        return img_list, label_list
###########################################################
'''
5.分割细节
'''
###########################################################
import shutil
import os.path as osp


def save_checkpoint(state, best_pred, epoch, is_best, checkpoint_path, filename='./checkpoint/checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, osp.join(checkpoint_path, 'best.pth'))
    # shutil.copyfile(filename, osp.join(checkpoint_path, 'model_ep{}.pth'.format(epoch+1)))


def adjust_learning_rate(opt, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    if opt.lr_mode == 'step':
        lr = opt.lr * (0.1 ** (epoch // opt.step))
    elif opt.lr_mode == 'poly':
        lr = opt.lr * (1 - epoch / opt.num_epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(opt.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def one_hot_it(label, label_info):
    # return semantic_map -> [H, W, num_classes]
    semantic_map = []
    for info in label_info:
        color = label_info[info]
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def compute_score(predict, target, forground=1, smooth=1):
    score = 0
    count = 0
    target[target != forground] = 0
    predict[predict != forground] = 0
    assert (predict.shape == target.shape)
    overlap = ((predict == forground) * (target == forground)).sum()  # TP
    union = (predict == forground).sum() + (target == forground).sum() - overlap  # FP+FN+TP
    FP = (predict == forground).sum() - overlap  # FP
    FN = (target == forground).sum() - overlap  # FN
    TN = target.shape[0] * target.shape[1] - union  # TN

    # print('overlap:',overlap)
    dice = (2 * overlap + smooth) / (union + overlap + smooth)

    precsion = ((predict == target).sum() + smooth) / (target.shape[0] * target.shape[1] + smooth)

    jaccard = (overlap + smooth) / (union + smooth)

    Sensitivity = (overlap + smooth) / ((target == forground).sum() + smooth)

    Specificity = (TN + smooth) / (FP + TN + smooth)

    return dice, precsion, jaccard, Sensitivity, Specificity


def eval_multi_seg(predict, target, num_classes):
    # pred_seg=torch.argmax(torch.exp(predict),dim=1).int()
    pred_seg = predict.data.cpu().numpy()
    label_seg = target.data.cpu().numpy().astype(dtype=np.int)
    assert (pred_seg.shape == label_seg.shape)
    acc = (pred_seg == label_seg).sum() / (pred_seg.shape[0] * pred_seg.shape[1] * pred_seg.shape[2])

    # Dice = []
    # Precsion = []
    # Jaccard = []
    # Sensitivity=[]
    # Specificity=[]

    # n = pred_seg.shape[0]
    Dice = []
    True_label = []
    TP = FPN = 0
    for classes in range(1, num_classes):
        overlap = ((pred_seg == classes) * (label_seg == classes)).sum()
        union = (pred_seg == classes).sum() + (label_seg == classes).sum()
        Dice.append((2 * overlap + 0.1) / (union + 0.1))
        True_label.append((label_seg == classes).sum())
        TP += overlap
        FPN += union

    return Dice, True_label, acc, 2 * TP / (FPN + 1)

    # for i in range(n):
    #     dice,precsion,jaccard,sensitivity,specificity= compute_score(pred_seg[i],label_seg[i])
    #     Dice.append(dice)
    #     Precsion .append(precsion)
    #     Jaccard.append(jaccard)
    #     Sensitivity.append(sensitivity)
    #     Specificity.append(specificity)

    # return Dice,Precsion,Jaccard,Sensitivity,Specificity


def eval_seg(predict, target, forground=1):
    pred_seg = torch.round(torch.sigmoid(predict)).int()
    pred_seg = pred_seg.data.cpu().numpy()
    label_seg = target.data.cpu().numpy().astype(dtype=np.int)
    assert (pred_seg.shape == label_seg.shape)

    Dice = []
    Precsion = []
    Jaccard = []
    n = pred_seg.shape[0]

    for i in range(n):
        dice, precsion, jaccard = compute_score(pred_seg[i], label_seg[i])
        Dice.append(dice)
        Precsion.append(precsion)
        Jaccard.append(jaccard)

    return Dice, Precsion, Jaccard


def batch_pix_accuracy(pred, label, nclass=1):
    if nclass == 1:
        pred = torch.round(torch.sigmoid(pred)).int()
        pred = pred.cpu().numpy()
    else:
        pred = torch.max(pred, dim=1)
        pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    pixel_labeled = np.sum(label >= 0)
    pixel_correct = np.sum(pred == label)

    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"

    return pixel_correct, pixel_labeled


def batch_intersection_union(predict, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int),note: not include background
    """
    if nclass == 1:
        pred = torch.round(torch.sigmoid(predict)).int()
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        area_inter = np.sum(pred * target)
        area_union = np.sum(pred) + np.sum(target) - area_inter

        return area_inter, area_union
###########################################################
'''
6.模型
'''
###########################################################
up_kwargs = {'mode': 'bilinear', 'align_corners': True}

#######################################################
'''
resnet
'''
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:57:49 2019

@author: Fsl
"""
import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append('/data/lihongxi/BA-Transformer/Cell-DETR-master')
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
# sys.path.insert(0, '../')

from models.utils.resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, \
    ResNet18_OS8, ResNet34_OS8
from models.utils.ASPP import ASPP, ASPP_Bottleneck

# class BAT(nn.Module):
#     def __init__(self,
#                  num_classes,
#                  num_layers,
#                  transformer_type_index=0,
#                  hidden_features=128,
#                  number_of_query_positions=1,
#                  segmentation_attention_heads=8):
#
#         super(BAT, self).__init__()
#
#         self.num_classes = num_classes
#         self.transformer_type = "BoundaryAwareTransformer" if transformer_type_index == 0 else "Transformer"
#         self.deeplab = base(num_classes, num_layers)
#         in_channels = 2048 if num_layers == 50 else 512
#
#
#
#         layers = num_layers
#         self.point_pred = point_pred
#
#         if layers == 18:
#             self.resnet = ResNet18_OS8()
#             self.aspp = ASPP(num_classes=self.num_classes)
#         elif layers == 50:
#             self.resnet = ResNet50_OS16()
#             self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)
#
#         self.transformer_attention_heads = transformer_attention_heads
#         self.num_encoder_layers = num_encoder_layers
#         self.num_decoder_layers = num_decoder_layers
#         self.hidden_features = hidden_features
#         self.number_of_query_positions = number_of_query_positions
#         self.transformer_activation = nn.LeakyReLU
#         self.segmentation_attention_heads = segmentation_attention_heads
#
#         in_channels = 2048 if layers == 50 else 512
#         self.convolution_mapping = nn.Conv2d(in_channels=in_channels,
#                                              out_channels=hidden_features,
#                                              kernel_size=(1, 1),
#                                              stride=(1, 1),
#                                              padding=(0, 0),
#                                              bias=True)
#
#         self.query_positions = nn.Parameter(data=torch.randn(
#             number_of_query_positions, hidden_features, dtype=torch.float),
#                                             requires_grad=True)
#
#         self.row_embedding = nn.Parameter(data=torch.randn(100,
#                                                            hidden_features //
#                                                            2,
#                                                            dtype=torch.float),
#                                           requires_grad=True)
#         self.column_embedding = nn.Parameter(data=torch.randn(
#             100, hidden_features // 2, dtype=torch.float),
#                                              requires_grad=True)
#
#         self.transformer = Transformer(d_model=hidden_features,
#                                        nhead=transformer_attention_heads,
#                                        num_encoder_layers=num_encoder_layers,
#                                        num_decoder_layers=num_decoder_layers,
#                                        dropout=dropout,
#                                        dim_feedforward=4 * hidden_features,
#                                        activation=self.transformer_activation)
#
#         self.trans_out_conv = nn.Conv2d(
#             hidden_features + segmentation_attention_heads, in_channels, 1, 1)
#
#         self.segmentation_attention_head = MultiHeadAttention(
#             query_dimension=hidden_features,
#             hidden_features=hidden_features,
#             number_of_heads=segmentation_attention_heads,
#             dropout=dropout)
#         # encoder后的feature首先输入到point_pre_layer
#         self.point_pre_layer = nn.Conv2d(hidden_features, 1, kernel_size=1)
#         print('dropout',dropout)
#
#     def forward(self, x):
#         # x (2,3,256,448)
#         h = x.size()[2]
#         w = x.size()[3]
#         feature_map = self.resnet(x) # feature_map (2,2048,16,28)
#         features = self.convolution_mapping(feature_map) # (2,128,16,28)
#         reshapef = nn.AvgPool2d((3,13), stride=(1,1),padding=(1,0))
#         features = reshapef(features) # (2,128,16,16)
#         height, width = features.shape[2:]
#         # 先定义x
#         batch_size = features.shape[0]
#         positional_embeddings = torch.cat([
#             self.column_embedding[:height].unsqueeze(dim=0).repeat(
#                 height, 1, 1),
#             self.row_embedding[:width].unsqueeze(dim=1).repeat(1, width, 1)
#         ],
#                                           dim=-1).permute(
#                                               2, 0, 1).unsqueeze(0).repeat(
#                                                   batch_size, 1, 1, 1)
#         boundary_embedding, features_encoded = self.transformer(
#             features, None, self.query_positions, positional_embeddings)
#         boundary_embedding = boundary_embedding.permute(2, 0, 1)
#
#         if self.point_pred == 1:
#             point_map = self.point_pre_layer(features_encoded)
#             point_map = torch.sigmoid(point_map)
#             features_encoded = point_map * features_encoded + features_encoded  #  Z=V+V*M
#
#         point_map_2 = self.segmentation_attention_head(
#             boundary_embedding, features_encoded.contiguous())
#
#         trans_feature_maps = torch.cat((features_encoded, point_map_2[:, 0]),
#                                        dim=1)
#         trans_feature_maps = self.trans_out_conv(trans_feature_maps)
#
#         output = self.aspp(
#             feature_map)  # (shape: (batch_size, num_classes, h/16, w/16))
#         output = F.interpolate(
#             output, size=(h, w),
#             mode="bilinear")  # (shape: (batch_size, num_classes, h, w))
#         # interpolate是上采样
#         main_out = F.log_softmax(output, dim=1)
#
#         if self.point_pred == 0:
#             return output
#
#         elif self.point_pred == 1:
#             # return output, point_map
#             return output, main_out


up_kwargs = {'mode': 'bilinear', 'align_corners': True}

#######################################################
'''
resnet
'''
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:57:49 2019

@author: Fsl
"""

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchsummary

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_base=False, stem_width=32):
        self.inplanes = stem_width * 2 if deep_base else 64

        super(ResNet, self).__init__()
        if deep_base:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        t = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return t


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print('使用了34')
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    model_dict = model.state_dict()

    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'],
                                             model_dir='/home/lihongxi/JBHI/Pretrain_model')  # Modify 'model_dir' according to your own path
        print('Petrain Model Have been loaded!')
        # pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'],
                                             model_dir='/data/lihongxi/model_weight')  # Modify 'model_dir' according to your own path
        print('Petrain Model Have been loaded!')
        # pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:

        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


# net = resnet34(pretrained=False)
# torchsummary.summary(net, (3, 512, 512))

class GPG_3(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(GPG_3, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(3 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))
        self.se = SELayer(3 * width, reduction=16)
        self.dilation1 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(
            SeparableConv2d(3 * width, width, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = self.se(feat)
        # feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat)], dim=1)
        feat = self.conv_out(feat)
        return feat


class GPG_4(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(GPG_4, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(2 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))
        self.se = SELayer(2 * width, reduction=16)
        self.dilation1 = nn.Sequential(
            SeparableConv2d(2 * width, width, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(
            SeparableConv2d(2 * width, width, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs): # expan[1] = 256 expan[2] = 512

        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2])] # conv5(512) conv4(256)
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = self.se(feat)
        # feat = torch.cat([self.dilation1(feat), self.dilation2(feat)], dim=1)
        feat = self.conv_out(feat)
        return feat
# SK模块

import torch
from torch import nn


# 被替换的3*3卷积
class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        self.gap = nn.AvgPool2d(int(WH / stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = self.gap(fea_U).squeeze_()
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


# 新的残差块结构
class SKUnit(nn.Module):
    def __init__(self, in_features, out_features, WH, M, G, r, mid_features=None, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features / 2)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1),
            nn.BatchNorm2d(mid_features),
            SKConv(mid_features, WH, M, G, r, stride=stride, L=L),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1, stride=1),
            nn.BatchNorm2d(out_features)
        )
        if in_features == out_features:  # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else:  # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d(out_features)
            )

    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)


class SKNet(nn.Module):
    def __init__(self, class_num):
        super(SKNet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )  # 32x32
        self.stage_1 = nn.Sequential(
            SKUnit(64, 256, 32, 2, 8, 2, stride=2),
            nn.ReLU(),
            SKUnit(256, 256, 32, 2, 8, 2),
            nn.ReLU(),
            SKUnit(256, 256, 32, 2, 8, 2),
            nn.ReLU()
        )  # 32x32
        self.stage_2 = nn.Sequential(
            SKUnit(256, 512, 32, 2, 8, 2, stride=2),
            nn.ReLU(),
            SKUnit(512, 512, 32, 2, 8, 2),
            nn.ReLU(),
            SKUnit(512, 512, 32, 2, 8, 2),
            nn.ReLU()
        )  # 16x16
        self.stage_3 = nn.Sequential(
            SKUnit(512, 1024, 32, 2, 8, 2, stride=2),
            nn.ReLU(),
            SKUnit(1024, 1024, 32, 2, 8, 2),
            nn.ReLU(),
            SKUnit(1024, 1024, 32, 2, 8, 2),
            nn.ReLU()
        )  # 8x8
        self.pool = nn.AvgPool2d(8)
        self.classifier = nn.Sequential(
            nn.Linear(1024, class_num),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.pool(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea


# if __name__ == '__main__':
#     x = torch.rand(8, 64, 32, 32)
#     conv = SKConv(64, 32, 3, 8, 2)
#     out = conv(x)
#     criterion = nn.L1Loss()
#     loss = criterion(out, x)
#     loss.backward()
#     print('out shape : {}'.format(out.shape))
#     print('loss value : {}'.format(loss))

# SE模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class GPG_2(nn.Module):
    def __init__(self, in_channels, width=512, up_kwargs=None, norm_layer=nn.BatchNorm2d):
        super(GPG_2, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[-4], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))

        self.conv_out = nn.Sequential(
            nn.Conv2d(4 * width, width, 1, padding=0, bias=False), # 将cat后的通道调整为原来的c2通道64
            nn.BatchNorm2d(width))
        self.se = SELayer(4 * width, reduction=16)
        self.dilation1 = nn.Sequential(
            SeparableConv2d(4 * width, width, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(
            SeparableConv2d(4 * width, width, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(
            SeparableConv2d(4 * width, width, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(
            SeparableConv2d(4 * width, width, kernel_size=3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, *inputs):
        # inputs = [c2,c3,c4,c5] 通过卷积使 c3,c4,c5的通道与c2对应  (保证H和W相同，通道不对应也没问题)
        # feats = [inputs[-1],inputs[-2],inputs[-3],inputs[-4]]
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3]), self.conv2(inputs[-4])]
        # feats = [c5(2,64,8,14), c4(2,64,16,28),c3(2,64,32,56),c2(2,64,64,112)]
        _, _, h, w = feats[-1].size() # (2,64,64,112)
        # 通过上采样使H和W与c2一致
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feats[-4] = F.interpolate(feats[-4], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1) # 通道堆叠 (2,256,64,112)
        # 加入channel注意力
        feat = self.se(feat)
        # feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)],
        #                  dim=1)  # (2,256,64,112)
        feat = self.conv_out(feat) # (2,64,64,112)
        return feat


class BaseNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BaseNetHead, self).__init__()
        if is_aux:
            self.conv_1x1_3x3 = nn.Sequential(
                ConvBnRelu(in_planes, 64, 1, 1, 0,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False),
                ConvBnRelu(64, 64, 3, 1, 1,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False))
        else:
            self.conv_1x1_3x3 = nn.Sequential(
                ConvBnRelu(in_planes, 32, 1, 1, 0,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False),
                ConvBnRelu(32, 32, 3, 1, 1,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False))
        # self.dropout = nn.Dropout(0.5)
        if is_aux:
            self.conv_1x1_2 = nn.Conv2d(64, out_planes, kernel_size=1,
                                        stride=1, padding=0)
        else:
            self.conv_1x1_2 = nn.Conv2d(32, out_planes, kernel_size=1,
                                        stride=1, padding=0)
        self.scale = scale

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale,
                              mode='bilinear',
                              align_corners=True)
        fm = self.conv_1x1_3x3(x)
        # fm = self.dropout(fm)
        output = self.conv_1x1_2(fm)
        return output


class SAPblock(nn.Module):
    def __init__(self, in_channels):
        super(SAPblock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, dilation=1, kernel_size=3,
                                 padding=1)

        self.bn = nn.ModuleList([nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels), nn.BatchNorm2d(in_channels)])
        self.conv1x1 = nn.ModuleList(
            [nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, dilation=1, kernel_size=1, padding=0),
             nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels, dilation=1, kernel_size=1, padding=0)])
        self.conv3x3_1 = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, dilation=1, kernel_size=3, padding=1),
             nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, dilation=1, kernel_size=3, padding=1)])
        self.conv3x3_2 = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels // 2, out_channels=2, dilation=1, kernel_size=3, padding=1),
             nn.Conv2d(in_channels=in_channels // 2, out_channels=2, dilation=1, kernel_size=3, padding=1)])
        self.conv_last = ConvBnRelu(in_planes=in_channels, out_planes=in_channels, ksize=1, stride=1, pad=0, dilation=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_size = x.size()

        branches_1 = self.conv3x3(x)
        branches_1 = self.bn[0](branches_1)

        branches_2 = F.conv2d(x, self.conv3x3.weight, padding=2, dilation=2)  # share weight
        branches_2 = self.bn[1](branches_2)

        branches_3 = F.conv2d(x, self.conv3x3.weight, padding=4, dilation=4)  # share weight
        branches_3 = self.bn[2](branches_3)

        feat = torch.cat([branches_1, branches_2], dim=1)
        # feat=feat_cat.detach()
        feat = self.relu(self.conv1x1[0](feat))
        feat = self.relu(self.conv3x3_1[0](feat))
        att = self.conv3x3_2[0](feat)
        att = F.softmax(att, dim=1)

        att_1 = att[:, 0, :, :].unsqueeze(1)
        att_2 = att[:, 1, :, :].unsqueeze(1)

        fusion_1_2 = att_1 * branches_1 + att_2 * branches_2

        feat1 = torch.cat([fusion_1_2, branches_3], dim=1)
        # feat=feat_cat.detach()
        feat1 = self.relu(self.conv1x1[0](feat1))
        feat1 = self.relu(self.conv3x3_1[0](feat1))
        att1 = self.conv3x3_2[0](feat1)
        att1 = F.softmax(att1, dim=1)

        att_1_2 = att1[:, 0, :, :].unsqueeze(1)
        att_3 = att1[:, 1, :, :].unsqueeze(1)

        ax = self.relu(self.gamma * (att_1_2 * fusion_1_2 + att_3 * branches_3) + (1 - self.gamma) * x)
        ax = self.conv_last(ax)

        return ax


class DecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d, scale=2, relu=True, last=False):
        super(DecoderBlock, self).__init__()

        self.conv_3x3 = ConvBnRelu(in_planes, in_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)

        self.sap = SAPblock(in_planes)
        self.scale = scale
        self.last = last

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.last == False:
            x = self.conv_3x3(x)
            # x=self.sap(x)
        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x = self.conv_1x1(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        inputs = inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
        inputs = inputs.view(in_size[0], in_size[1], 1, 1)

        return inputs
class jiangwei_CNN(nn.Module):
    def __init__(self,in_channels,out_channels):
        """
        创建一个卷积神经网络
        网络只有两层
        :param in_channels: 输入通道数量
        :param out_channels: 输出通道数量
        """
        super(jiangwei_CNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,1,stride=1,padding=0)
        # self.pool1=nn.MaxPool2d(kernel_size=2,stride=1)
        # self.conv2=nn.Conv2d(10,out_channels,3,stride=1,padding=1)
        # self.pool2=nn.MaxPool2d(kernel_size=2,stride=1)
    def forward(self,x):
        """
        前向传播函数
        :param x:  输入，tensor 类型
        :return: 返回结果
        """
        out=self.conv1(x)
        # out=self.pool1(out)
        # out=self.conv2(out)
        # out=self.pool2(out)
        return out


class CPFNet(nn.Module):
    def __init__(self,
                 out_planes=2,
                 num_layers= 50,
                 transformer_type_index=0,# 50
                 point_pred = 1, # 1
                 ccm=True,
                 norm_layer=nn.BatchNorm2d,
                 is_training=True,
                 expansion=2,
                 base_channel=32,
                 hidden_features=128,
                 number_of_query_positions=1,
                 transformer_attention_heads=8,
                 num_encoder_layers=3,
                 num_decoder_layers=2,
                 segmentation_attention_heads=8,
                 dropout=0,):
        super(CPFNet, self).__init__()

        self.num_classes = out_planes

        self.transformer_type = "BoundaryAwareTransformer" if transformer_type_index == 0 else "Transformer"

        self.deeplab = base(out_planes, num_layers)

        in_channels = 512 # //



        layers = num_layers
        self.point_pred = point_pred

        if layers == 18:
            self.resnet = ResNet18_OS8()
            self.aspp = ASPP(num_classes=self.num_classes)
        elif layers == 50:
            self.resnet = ResNet50_OS16()
            self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)

        # CPF
        self.backbone = resnet50(pretrained=False)
        self.expansion = expansion
        self.base_channel = base_channel
        if self.expansion == 4 and self.base_channel == 64:
            expan = [512, 1024, 2048]
            spatial_ch = [128, 256]
        elif self.expansion == 4 and self.base_channel == 32:
            expan = [256, 512, 1024]
            spatial_ch = [32, 128]
            conv_channel_up = [256, 384, 512]
        elif self.expansion == 2 and self.base_channel == 32: # use it
            expan = [128, 256, 512]
            spatial_ch = [64, 64]
            conv_channel_up = [128, 256, 512]
        self.jiangwei256_64 = jiangwei_CNN(in_channels=256, out_channels=64)
        self.jiangwei512_128 = jiangwei_CNN(in_channels=512, out_channels=128)
        self.jiangwei1024_256 = jiangwei_CNN(in_channels=1024, out_channels=256)
        self.jiangwei2048_512 = jiangwei_CNN(in_channels=2048, out_channels=512)
        conv_channel = expan[0]
        self.is_training = is_training
        self.sap = SAPblock(expan[-1])

        # BAT
        self.transformer_attention_heads = transformer_attention_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.hidden_features = hidden_features
        self.number_of_query_positions = number_of_query_positions
        self.transformer_activation = nn.LeakyReLU
        self.segmentation_attention_heads = segmentation_attention_heads
        # in_channels = 2048 if layers == 50 else 512
        self.convolution_mapping = nn.Conv2d(in_channels= 512,
                                             out_channels=hidden_features,
                                             kernel_size=(1, 1),
                                             stride=(1, 1),
                                             padding=(0, 0),
                                             bias=True)
        self.query_positions = nn.Parameter(data=torch.randn(
            number_of_query_positions, hidden_features, dtype=torch.float),
            requires_grad=True)
        self.column_embedding = nn.Parameter(data=torch.randn(
            100, hidden_features // 2, dtype=torch.float),
            requires_grad=True)
        self.row_embedding = nn.Parameter(data=torch.randn(100,
                                                           hidden_features //
                                                           2,
                                                           dtype=torch.float),
                                          requires_grad=True)
        # loading Cell_DETR weights
        self.transformer = cell_detr_128(pretrained=False, type_index=transformer_type_index)

        # self.trans_out_conv = nn.Conv2d(
        #     hidden_features + segmentation_attention_heads, 512, 1, 1)
        self.trans_out_conv = nn.Conv2d(
            128, 512, 1, 1)
        self.segmentation_attention_head = MultiHeadAttention(
            query_dimension=hidden_features,
            hidden_features=hidden_features,
            number_of_heads=segmentation_attention_heads,
            dropout=0)
        # encoder后的feature首先输入到point_pre_layer
        self.point_pre_layer = nn.Conv2d(hidden_features, 1, kernel_size=1)

        self.decoder5 = DecoderBlock(expan[-1], expan[-2], relu=False, last=True)  # 256
        self.decoder4 = DecoderBlock(expan[-2], expan[-3], relu=False)  # 128
        self.decoder3 = DecoderBlock(expan[-3], spatial_ch[-1], relu=False)  # 64
        self.decoder2 = DecoderBlock(spatial_ch[-1], spatial_ch[-2])  # 32
        self.mce_2 = GPG_2([spatial_ch[-1], expan[0], expan[1], expan[2]], width=spatial_ch[-1], up_kwargs=up_kwargs) # 64,128,256,512 width =64 也就是各个特征经过卷积后的channel为64
        self.mce_3 = GPG_3([expan[0], expan[1], expan[2]], width=expan[0], up_kwargs=up_kwargs)
        self.mce_4 = GPG_4([expan[1], expan[2]], width=expan[1], up_kwargs=up_kwargs) # expan[1] = 256 expan[2] = 512

        self.main_head = BaseNetHead(spatial_ch[0], out_planes, 2,
                                     is_aux=False, norm_layer=norm_layer)
        self.relu = nn.ReLU()

    def forward(self, x): # x=(2,3,256,448)
        x = self.backbone.conv1(x) # x = (2，64，128，224) 【3, 64, kernel_size=7, stride=2, padding=3】 s为2，减少了1/2
        x = self.backbone.bn1(x)
        c1 = self.backbone.relu(x)  # 1/2(原来大小的)  64
        x = self.backbone.maxpool(c1) # x = (2,64,64,112)
        c2 = self.backbone.layer1(x)  # 1/4   64 (2,64,64,112)
        # print('c2维度',c2.shape)
        c_2 = self.jiangwei256_64(c2)
        c3 = self.backbone.layer2(c2)  # 1/8   128  (2,128,32,56)
        c_3 = self.jiangwei512_128(c3)
        c4 = self.backbone.layer3(c3)  # 1/16   256  (2,256,16,28)
        c_4 = self.jiangwei1024_256(c4)
        c5 = self.backbone.layer4(c4)  # 1/32   512  (2,512,8,14)
        # print('c5维度',c5.shape)
        c_5 = self.jiangwei2048_512(c5)
        feature_map = c_5
        # features = self.convolution_mapping(feature_map)
        #  c5对应featuremap,在c5之后加入transformer
        reshapef = nn.AvgPool2d((1,7), stride=(1,1),padding=(0,0))
        feature_map1 = reshapef(feature_map)
        # features = reshapef(features)  # (2,512,8,8)
        b, _, h, w = feature_map.size()

        features = self.convolution_mapping(feature_map)
        features = reshapef(features)  # (2,128,16,16)
        h, w = features.shape[2:]
        # height, width = features.shape[2:]
        # batch_size = features.shape[0]
        # positional_embeddings = torch.cat([
        #     self.column_embedding[:height].unsqueeze(dim=0).repeat(
        #         height, 1, 1),
        #     self.row_embedding[:width].unsqueeze(dim=1).repeat(1, width, 1)
        # ],
        #     dim=-1).permute(
        #     2, 0, 1).unsqueeze(0).repeat(
        #     batch_size, 1, 1, 1)
        # # (2,128,8,8)
        # boundary_embedding, features_encoded = self.transformer(
        #     features, None, self.query_positions, positional_embeddings)
        # boundary_embedding = boundary_embedding.permute(2, 0, 1) # (2,1,128)
        # if self.point_pred == 1:
        #     point_map = self.point_pre_layer(features_encoded) # (2,1,8,8)
        #     point_map = torch.sigmoid(point_map)
        #     features_encoded = point_map * features_encoded + features_encoded  #  Z=V+V*M
        #
        # point_map_2 = self.segmentation_attention_head(
        #     boundary_embedding, features_encoded.contiguous()) # (2,1,8,8,8)
        d = self.column_embedding  # (100,64)
        print('打印d的维度',d)
        dun = d[:h] # (8,64)
        print('打印dun的维度', dun.shape)
        t = self.column_embedding[:h].unsqueeze(dim=1).repeat(1, w, 1)  # repeat之前(1，8，64)之后(8,8,64)
        print('打印t的维度', t.shape)
        positional_embeddings = torch.cat(
            [self.column_embedding[:h].unsqueeze(dim=0).repeat(h, 1, 1),
             self.row_embedding[:w].unsqueeze(dim=1).repeat(1, w, 1)],
            dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1) # (2,128,8,8)
        print('positional_embeddings', positional_embeddings.shape)

        # print('positional_embeddings',positional_embeddings.shape)
        if self.transformer_type == 'BoundaryAwareTransformer':

            latent_tensor, features_encoded, point_maps = self.transformer(
                features, None, self.query_positions, positional_embeddings)
        else:
            # 该代码没有point
            latent_tensor, features_encoded = self.transformer(
                features, None, self.query_positions, positional_embeddings)
            point_maps = []
        latent_tensor = latent_tensor.permute(2, 0, 1)
        point_dec = self.segmentation_attention_head(
            latent_tensor, features_encoded.contiguous())
        features_encoded = point_dec * features_encoded + features_encoded # (8, 128, 8, 8)
        point_maps.append(point_dec)
        # print('features_encoded',features_encoded.shape)
        trans_feature_maps = self.trans_out_conv(features_encoded) # 原来的(2,512,8,8) 现在的[8, 512, 8, 8]
        # print('trans_feature_maps相加前',trans_feature_maps.shape)
        trans_feature_maps = trans_feature_maps + feature_map1 # transformer的结果与原featuremap融合 现在的 [8, 512, 8, 8
        # print('trans_feature_maps相加后', trans_feature_maps.shape)
        # transfomer生成pointout pre 而output相当于传统的UNNET
        # 此处可以不经过aspp，可以做一组实验对比
        # output = self.deeplab.aspp(
        #     trans_feature_maps)  # (shape: (batch_size, num_classes, h/16, w/16)) (2，2，8，14)
        print('trans_feature_maps', trans_feature_maps.shape)  # (n,512,8,8)
        output = F.interpolate(
            trans_feature_maps, size=(8, 14),
            mode="bilinear")  # (shape: (batch_size, num_classes, h, w)) (2,2,256,448)

        # interpolate是上采样
        # main_out = F.log_softmax(output, dim=1)
        # 维度恢复为c5的维度

        m2 = self.mce_2(c_2, c_3, c_4, c_5) # (2,64,64,,112)
        m3 = self.mce_3(c_3, c_4, c_5) # m3 (2,128,32,56)
        m4 = self.mce_4(c_4, c_5) # c4 (2,256,16,28)
        # c5 = self.sap(c5) # c5 (2,512,8,14)
        c5 = output
        print('c5',c5.shape)
        d4 = self.relu(self.decoder5(c5) + m4)  # 256 d4=(2,256,16,28)
        d3 = self.relu(self.decoder4(d4) + m3)  # 128 d3 =(2,128,32,56)
        d2 = self.relu(self.decoder3(d3) + m2)  # 64 (2,64,64,112)
        d1 = self.decoder2(d2) + c1  # 32  (2,64,128,224)
        main_out = self.main_head(d1)  # (2,2,256,448)
        main_out = F.log_softmax(main_out, dim=1)
        return main_out, main_out # 第一个是aux_out，第二个是main_out(经过了softmax)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)




###########################################################
'''
8.模型训练
'''
###########################################################
def val(args, model, dataloader):
    print('\n')
    print('Start Validation!')
    with torch.no_grad():
        model.eval()
        tbar = tqdm.tqdm(dataloader, desc='\r')

        total_Dice = []
        total_Dice1 = []
        total_Dice2 = []
        total_Dice3 = []
        total_Dice.append(total_Dice1)
        total_Dice.append(total_Dice2)
        total_Dice.append(total_Dice3)
        Acc = []

        cur_cube = []
        cur_label_cube = []
        next_cube = []
        counter = 0
        end_flag = False

        for i, (data, labels) in enumerate(tbar):
            # tbar.update()
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = labels[0].cuda()
            slice_num = labels[1][0].long().item()

            # get RGB predict image

            aux_predict, predicts = model(data)

            predict = torch.argmax(torch.exp(aux_predict), dim=1)
            batch_size = predict.size()[0]

            counter += batch_size
            if counter <= slice_num:
                cur_cube.append(predict)
                cur_label_cube.append(label)
                if counter == slice_num:
                    end_flag = True
                    counter = 0
            else:
                last = batch_size - (counter - slice_num)

                last_p = predict[0:last]
                last_l = label[0:last]

                first_p = predict[last:]
                first_l = label[last:]

                cur_cube.append(last_p)
                cur_label_cube.append(last_l)
                end_flag = True
                counter = counter - slice_num

            if end_flag:
                end_flag = False
                predict_cube = torch.stack(cur_cube, dim=0).squeeze()
                label_cube = torch.stack(cur_label_cube, dim=0).squeeze()
                cur_cube = []
                cur_label_cube = []
                if counter != 0:
                    cur_cube.append(first_p)
                    cur_label_cube.append(first_l)

                assert predict_cube.size()[0] == slice_num
                Dice, true_label, acc, mean_dice = eval_multi_seg(predict_cube, label_cube, args.num_classes)

                for class_id in range(args.num_classes - 1):
                    if true_label[class_id] != 0:
                        total_Dice[class_id].append(Dice[class_id])
                Acc.append(acc)
                len0 = len(total_Dice[0]) if len(total_Dice[0]) != 0 else 1
                len1 = len(total_Dice[1]) if len(total_Dice[1]) != 0 else 1
                len2 = len(total_Dice[2]) if len(total_Dice[2]) != 0 else 1

                dice1 = sum(total_Dice[0]) / len0
                dice2 = sum(total_Dice[1]) / len1
                dice3 = sum(total_Dice[2]) / len2
                ACC = sum(Acc) / len(Acc)
                tbar.set_description('Mean_D: %3f, Dice1: %.3f, Dice2: %.3f, Dice3: %.3f, ACC: %.3f' % (
                mean_dice, dice1, dice2, dice3, ACC))
        print('Mean_Dice:', mean_dice)
        print('Dice1:', dice1)
        print('Dice2:', dice2)
        print('Dice3:', dice3)
        print('Acc:', ACC)
        return mean_dice, dice1, dice2, dice3, ACC

def CE_Loss(inputs, target, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss = nn.NLLLoss(ignore_index=num_classes)(F.log_softmax(temp_inputs, dim=-1), temp_target)
    return CE_loss
# criterion = SoftDiceLoss()
def train(args, model, optimizer, criterion, dataloader_train, dataloader_val, ):
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(args.log_dirs, current_time + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)
    step = 0
    best_pred = 0.0

    # print('在第0维插入', torch.stack([c, d], dim=0))
    for epoch in range(args.num_epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        train_loss = 0.0
        #        is_best=False
        for i, (data, label) in enumerate(dataloader_train):
            # if i>9:
            #     break
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda().long()
            optimizer.zero_grad()
            aux_out, main_out = model(data)
            # print('label', label.shape)
            # print('data', data.shape)
            c = torch.rand(3, 4, 5)
            d = torch.rand(3, 4, 5)
            # label0 = torch.ones(2,256,448).cuda()-label
            # label1 = label
            # labela = torch.stack([label0, label1], dim=1) # [2, 2, 256, 448]
            # print('****target_soft',labela.shape)
            # target_soft = get_soft_label(labela, 2)  # get soft label
            # print('****target_soft',target_soft.shape)

            # label的shape(2,256,448) data的shape(2,3,256,448)
            # loss_aux = F.nll_loss(aux_out, label, weight=None)
            # mainout与label对不上
            # loss_main = criterion[1](aux_out, label)
            # label2 = label.unsqueeze(1)
            # print('**label2',label2.shape) # label2 (2,1,256,448)
            # label1和label2在第1维度上拼接
            # torch.cat(label1,dim=1)

            # loss = criterion(aux_out, target_soft, args.num_classes)
            loss_main = criterion[1](main_out, label)
            loss_CE = CE_Loss(aux_out,label,num_classes= args.num_classes)
            loss = loss_CE + loss_main
            loss.backward()
            optimizer.step()
            tq.update(args.batch_size)
            train_loss += loss.item()
            tq.set_postfix(loss='%.6f' % (train_loss / (i + 1)))
            step += 1
            if step % 10 == 0:
                writer.add_scalar('Train/loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('Train/loss_epoch', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))

        if epoch % args.validation_step == 0:
            print('one epoch over')
            # mean_Dice, Dice1, Dice2, Dice3, acc = val(args, model, dataloader_val)
            # writer.add_scalar('Valid/Mean_val', mean_Dice, epoch)
            # writer.add_scalar('Valid/Dice1_val', Dice1, epoch)
            # writer.add_scalar('Valid/Dice2_val', Dice2, epoch)
            # writer.add_scalar('Valid/Dice3_val', Dice3, epoch)
            # writer.add_scalar('Valid/Acc_val', acc, epoch)
            # # mean_Dice=(Dice1+Dice2+Dice3)/3.0
            # is_best = mean_Dice > best_pred
            # best_pred = max(best_pred, mean_Dice)
            # print(best_pred)
            checkpoint_dir = args.save_model_path
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_latest = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_dice': best_pred,
            }, best_pred, epoch, 1, checkpoint_dir, filename=checkpoint_latest) # 源代码中is_best改成了1
import re
import  datetime
def test(model, dataloader, args, save_path):
    print('start test!')
    with torch.no_grad():
        model.eval()
        # precision_record = []
        tq = tqdm.tqdm(dataloader, desc='\r')
        tq.set_description('test')
        # comments就是当前的工作目录
        comments = os.getcwd().split('\\')[-1]
        for i, (data, label_path) in enumerate(tq):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                # label = label.cuda()
            # 模型返回两个值
            aux_pred, predict = model(data)
            predict = torch.argmax(torch.exp(aux_pred), dim=1)
            pred = predict.data.cpu().numpy()
            pred_RGB = Data.COLOR_DICT[pred.astype(np.uint8)]
            pred_RGB = torch.from_numpy(pred_RGB)
            k = pred_RGB.cpu().detach().numpy()
            res = k[0]  # 得到batch中其中一步的图片
            image = Image.fromarray(np.uint8(res)).convert('RGB')

            k = label_path[0]
            # 通过时间命名存储结

            pattern = re.compile(r'[0-9]*.png')
            str1 = k
            j = pattern.search(str1).group()
            savepath = str(j)
            predictpng_dir = 'predict_png/'
            pngsave_path = os.path.join(predictpng_dir, savepath)
            print('图片序号',j)
            image.save(pngsave_path)
        files_mask2rle('predict_png')
        #     for index, item in enumerate(label_path):
        #         img = Image.fromarray(pred_RGB[index].squeeze().astype(np.uint8))
        #         _, name = os.path.split(item)
        #
        #         img.save(os.path.join(save_path, name))
        #         # tq.set_postfix(str=str(save_img_path))
        # tq.close()


def main(mode='train', args=None):
    # create dataset and dataloader
    dataset_path = args.data
    dataset_train = Data(os.path.join(dataset_path, 'train'), scale=(args.crop_width, args.crop_height), mode='train')
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    dataset_val = Data(os.path.join(dataset_path, 'val'), scale=(args.crop_height, args.crop_width), mode='val')
    dataloader_val = DataLoader(
        dataset_val,
        # this has to be 1
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    dataset_test = Data(os.path.join(dataset_path, 'test'), scale=(args.crop_height, args.crop_width), mode='test')
    dataloader_test = DataLoader(
        dataset_test,
        # this has to be 1
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    # load modeL
    #1 model_path = r'checkpoints/bat475.pth'
    #
    model_all = {'BaseNet': CPFNet(out_planes=args.num_classes)}
    model = model_all[args.net_work]
    print(args.net_work)
    cudnn.benchmark = True
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    #2 checkpoint = torch.load(model_path)
    #3 model.load_state_dict(checkpoint['state_dict'])
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion_aux = nn.NLLLoss(weight=None)
    criterion_main = Multi_DiceLoss(class_num=args.num_classes)
    criterion = [criterion_aux, criterion_main]
    if mode == 'train':
        train(args, model, optimizer, criterion, dataloader_train, dataloader_val)
###########################################################
'''
9.主函数
'''
##########################################################
if __name__ == '__main__':
    print('不经过aspp')
    seed=1234
    torch.manual_seed(seed)   # 固定初始化
    torch.cuda.manual_seed_all(seed)
    args=DefaultConfig()
    modes = 'train'
    if modes=='train':
        main(mode='train', args=args)

# if __name__ == '__main__':
#     print('start test!')
#     save_path = r'test'
#     model_path = r'check_aCPF_BAT/checkpoint_latest.pth'
#     dataset_test = Data(os.path.join(DefaultConfig.data, 'test'), scale=(DefaultConfig.crop_width,
#                                                                          DefaultConfig.crop_height), mode='test')
#     args = DefaultConfig()
#     dataloader_test = DataLoader(
#         dataset_test,
#         batch_size=1,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=True,
#         drop_last=False
#     )
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
#     model_all = {'BaseNet': CPFNet(out_planes=args.num_classes)}
#     model = model_all[args.net_work]
#     cudnn.benchmark = True
#     if torch.cuda.is_available() and args.use_gpu:
#         model = torch.nn.DataParallel(model).cuda()
#     checkpoint = torch.load(model_path)
#     model.load_state_dict(checkpoint['state_dict'])
#     test(model, dataloader_test, args, save_path)