#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from models.util import get_func


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, func, inplanes, planes, stride=1, num_bit=1, wgt_sigma=1, wgt_temp=2, act_sigma=2, act_temp=2):
#         super(BasicBlock, self).__init__()
#         self.conv1 = func(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, num_bit=num_bit,
#                           wgt_sigma=wgt_sigma, wgt_temp=wgt_temp, act_sigma=act_sigma, act_temp=act_temp)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = func(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, num_bit=num_bit,
#                           wgt_sigma=wgt_sigma, wgt_temp=wgt_temp, act_sigma=act_sigma, act_temp=act_temp)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.shortcut = nn.Sequential()
#         if stride != 1 or inplanes != planes:
#             self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4),
#                                                         "constant", 0))
#
#     def forward(self, x):
#         conv1_out = F.relu(self.bn1(self.conv1(x)))
#         conv2_out = self.bn2(self.conv2(conv1_out))
#         out = conv2_out + self.shortcut(x)
#         out = F.relu(out)
#         return out, conv1_out, conv2_out
#
#
# class ResNet(nn.Module):
#
#     def __init__(self, block, num_blocks, num_classes=10, num_bit=1, wgt_sigma=1, wgt_temp=2, act_sigma=2, act_temp=2):
#         super(ResNet, self).__init__()
#         self.in_planes = 16
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, num_bit=num_bit,
#                                        wgt_sigma=wgt_sigma, wgt_temp=wgt_temp, act_sigma=act_sigma, act_temp=act_temp)
#         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, num_bit=num_bit,
#                                        wgt_sigma=wgt_sigma, wgt_temp=wgt_temp, act_sigma=act_sigma, act_temp=act_temp)
#         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, num_bit=num_bit,
#                                        wgt_sigma=wgt_sigma, wgt_temp=wgt_temp, act_sigma=act_sigma, act_temp=act_temp)
#
#         self.bn2 = nn.BatchNorm1d(64)
#
#         self.linear = nn.Linear(64, num_classes)
#
#     def _make_layer(self, block, planes, num_blocks, stride, num_bit, wgt_sigma, wgt_temp, act_sigma, act_temp):
#         strides = [stride] + [1]*(num_blocks-1)
#         ret_dict = dict()
#
#         for i, stride in enumerate(strides):
#             layers = []
#             layers.append(block(self.in_planes, planes, stride, num_bit, wgt_sigma, wgt_temp, act_sigma, act_temp))
#             ret_dict['block_{}'.format(i)] = nn.Sequential(*layers)
#             self.in_planes = planes * block.expansion
#
#         return nn.Sequential(OrderedDict(ret_dict))
#
#     def forward(self, x):
#         ret_dict = dict()
#         out = F.relu(self.conv1(x))
#         layer_names = self.layer1._modules.keys()
#         for i, layer_name in enumerate(layer_names):
#             out, conv1_out, conv2_out = self.layer1._modules[layer_name](out)
#             ret_dict['layer1_{}_conv1'.format(i)] = conv1_out
#             ret_dict['layer1_{}_conv2'.format(i)] = conv2_out
#
#         layer_names = self.layer2._modules.keys()
#         for i, layer_name in enumerate(layer_names):
#             out, conv1_out, conv2_out = self.layer2._modules[layer_name](out)
#             ret_dict['layer2_{}_conv1'.format(i)] = conv1_out
#             ret_dict['layer2_{}_conv2'.format(i)] = conv2_out
#
#         layer_names = self.layer3._modules.keys()
#         for i, layer_name in enumerate(layer_names):
#             out, conv1_out, conv2_out = self.layer3._modules[layer_name](out)
#             ret_dict['layer3_{}_conv1'.format(i)] = conv1_out
#             ret_dict['layer3_{}_conv2'.format(i)] = conv2_out
#
#         out = F.avg_pool2d(out, out.size()[3])
#         out = out.view(out.size(0), -1)
#         out = self.bn2(out)
#         out = self.linear(out)
#         ret_dict['out'] = out
#         return ret_dict


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, func, params, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        conv = get_func(func)
        self.conv1 = conv(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, **params)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, **params)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4),
                                                        "constant", 0))

    def forward(self, x):
        conv1_out = F.relu(self.bn1(self.conv1(x)))
        conv2_out = self.bn2(self.conv2(conv1_out))
        out = conv2_out + self.shortcut(x)
        out = F.relu(out)
        return out, conv1_out, conv2_out


class ResNet(nn.Module):

    def __init__(self, func, params, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(func, params, block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(func, params, block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(func, params, block, 64, num_blocks[2], stride=2)

        self.bn2 = nn.BatchNorm1d(64)

        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, func, params, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        ret_dict = dict()
        for i, stride in enumerate(strides):
            layers = []
            layers.append(block(func, params, self.in_planes, planes, stride))
            ret_dict['block_{}'.format(i)] = nn.Sequential(*layers)
            self.in_planes = planes * block.expansion

        return nn.Sequential(OrderedDict(ret_dict))

    def forward(self, x):
        ret_dict = dict()
        out = F.relu(self.conv1(x))
        layer_names = self.layer1._modules.keys()
        for i, layer_name in enumerate(layer_names):
            out, conv1_out, conv2_out = self.layer1._modules[layer_name](out)
            ret_dict['layer1_{}_conv1'.format(i)] = conv1_out
            ret_dict['layer1_{}_conv2'.format(i)] = conv2_out

        layer_names = self.layer2._modules.keys()
        for i, layer_name in enumerate(layer_names):
            out, conv1_out, conv2_out = self.layer2._modules[layer_name](out)
            ret_dict['layer2_{}_conv1'.format(i)] = conv1_out
            ret_dict['layer2_{}_conv2'.format(i)] = conv2_out

        layer_names = self.layer3._modules.keys()
        for i, layer_name in enumerate(layer_names):
            out, conv1_out, conv2_out = self.layer3._modules[layer_name](out)
            ret_dict['layer3_{}_conv1'.format(i)] = conv1_out
            ret_dict['layer3_{}_conv2'.format(i)] = conv2_out

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.linear(out)
        ret_dict['out'] = out
        return ret_dict


def resnet20(quant_func, quant_params, **kwargs):
    """ResNet-20 model.
    """
    print(kwargs)
    return ResNet(quant_func, quant_params, BasicBlock, [3, 3, 3], **kwargs)
