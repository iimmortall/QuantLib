#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
import pdb
import numpy as np
from collections import OrderedDict

__all__ = ['ResNet', 'resnet20_daq']


import torch
import torch.nn as nn
import torch.nn.functional as F


class absol(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.abs()

    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.sign(input)
        grad_input = grad_input + 1
        grad_input = ((grad_input+1e-6)/2).round()
        grad_input = (2*grad_input) - 1
        return grad_output * grad_input


class QConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_bit=1, wgt_sigma=1, wgt_temp=2, act_sigma=2, act_temp=2, quant_act=True, is_quant=True):
        super(QConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_bit = num_bit
        self.quant_act = quant_act
        self.bit_range = 2**self.num_bit - 1
        self.is_quant = is_quant
        self.q_value = torch.from_numpy(np.linspace(0, 1, 2))
        self.q_value = self.q_value.reshape(len(self.q_value), 1, 1, 1, 1).float().cuda()
        self.wgt_sigma = wgt_sigma
        self.wgt_temp = wgt_temp
        self.act_sigma = act_sigma
        self.act_temp = act_temp

        if self.is_quant:
            # using int32 max/min as init and back propagation to optimization
            # Weight
            self.uW = nn.Parameter(data=torch.tensor(2**31 - 1).float().cuda())
            self.lW = nn.Parameter(data=torch.tensor((-1) * (2**32)).float().cuda())
            self.register_buffer('init', torch.tensor(1).float().cuda())
            self.beta = nn.Parameter(data=torch.tensor(0.2).float().cuda())

            # Activation input
            if self.quant_act:
                self.uA = nn.Parameter(data=torch.tensor(2**31 - 1).float().cuda())

    def clipping(self, x, upper, lower):
        # clip lower
        x = x + F.relu(lower - x)
        # clip upper
        x = x - F.relu(x - upper)
        return x

    def w_quan(self, x, u, l):
        # For reducing inference time
        x = self.clipping(x, u, l)
        delta = (u - l) / self.bit_range
        interval = (x - l) / delta
        x_floor = interval.floor()
        interval = interval - x_floor
        output = 2*(interval.round() + x_floor) - self.bit_range
        return output / self.bit_range

    def w_soft_quan(self, x, u, l):
        delta = (u - l) / self.bit_range
        interval = (x - l) / delta
        interval = torch.clamp(interval, min=0, max=self.bit_range)
        output = 2 * self.w_soft_argmax(interval, self.wgt_temp, self.wgt_sigma) - self.bit_range
        return output / self.bit_range

    def w_soft_argmax(self, x, T, sigma):
        x_floor = x.floor()
        x = x - x_floor.detach()

        m_p = torch.exp(-absol.apply(x.unsqueeze(0).repeat(len(self.q_value.cuda()), 1, 1, 1, 1) - self.q_value.cuda()))

        # Get the kernel value
        max_value, max_idx = m_p.max(dim=0)
        max_idx = max_idx.unsqueeze(0).float().cuda()
        k_p = torch.exp(-(torch.pow(self.q_value.cuda()-max_idx, 2).float()/(sigma**2)))

        # Get the score
        score = m_p * k_p

        # Flexible temperature
        denorm = (score[0] - score[1]).abs()
        T_ori = T
        T = T / denorm
        T = T.detach()

        tmp_score = T * score

        # weighted average using the score and temperature
        prob = torch.exp(tmp_score - tmp_score.max())
        denorm2 = prob.sum(dim=0, keepdim=True)
        prob = prob / denorm2

        q_var = self.q_value.clone()
        q_var[0] = q_var[0] - (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        q_var[1] = q_var[1] + (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        output = (q_var * prob).sum(dim=0)
        output = output + x_floor

        return output

    def a_quan(self, x, u, l):
        # For reducing inference time
        x = self.clipping(x, u, l)
        delta = (u - l) / self.bit_range
        interval = (x - l) / delta
        x_floor = interval.floor()
        interval = interval - x_floor
        output = interval.round() + x_floor
        return output / self.bit_range

    def a_soft_quan(self, x, u, l):
        delta = (u - l) / self.bit_range
        interval = (x - l) / delta
        interval = torch.clamp(interval, min=0, max=self.bit_range)
        output = self.a_soft_argmax(interval, self.act_temp, self.act_sigma)
        return output / self.bit_range

    def a_soft_argmax(self, x, T, sigma):
        x_floor = x.floor()
        x = x - x_floor.detach()
        m_p = torch.exp(-absol.apply(x.unsqueeze(0).repeat(len(self.q_value), 1, 1, 1, 1) - self.q_value))

        # Get the kernel value
        max_value, max_idx = m_p.max(dim=0)
        max_idx = max_idx.unsqueeze(0).float().cuda()
        k_p = torch.exp(-(torch.pow(self.q_value-max_idx, 2).float()/(sigma**2)))

        # Get the score
        score = m_p * k_p

        # Flexible temperature
        denorm = (score[0] - score[1]).abs()
        T_ori = T
        T = T / denorm
        T = T.detach()

        tmp_score = T * score

        # weighted average using the score and temperature
        prob = torch.exp(tmp_score - tmp_score.max())
        denorm2 = prob.sum(dim=0, keepdim=True)
        prob = prob / denorm2

        q_var = self.q_value.clone()
        q_var[0] = q_var[0] - (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        q_var[1] = q_var[1] + (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        output = (q_var * prob).sum(dim=0)
        output = output + x_floor

        return output

    def forward(self, x):
        if self.is_quant:
            if self.init:
                self.lW.data = torch.tensor(-3.0).cuda()
                self.uW.data = torch.tensor(3.0).cuda()
                self.uA.data = x.std() * 3

            curr_running_lw = self.lW
            curr_running_uw = self.uW

            curr_running_la = 0
            curr_running_ua = self.uA

            # Weight normalization
            mean = self.weight.data.mean().cuda()
            std = self.weight.data.std().cuda()
            norm_weight = self.weight.add(-mean).div(std)

            # Weight quantization
            if self.training:
                Qweight = self.w_soft_quan(norm_weight, curr_running_uw, curr_running_lw)
            else:
                Qweight = self.w_quan(norm_weight, curr_running_uw, curr_running_lw)
            Qbias = self.bias

            # Activation quantization
            Qactivation = x
            if self.quant_act:
                if self.training:
                    Qactivation = self.a_soft_quan(x, curr_running_ua, curr_running_la)
                else:
                    Qactivation = self.a_quan(x, curr_running_ua, curr_running_la)

            if self.init == 1:
                # scale factor initialization
                q_output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
                ori_output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

                self.beta.data = torch.mean(torch.abs(ori_output)) / torch.mean(torch.abs(q_output))
                self.init = torch.tensor(0)

            output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
            output = torch.abs(self.beta) * output
        else:
            output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, num_bit=1, wgt_sigma=1, wgt_temp=2, act_sigma=2, act_temp=2):
        super(BasicBlock, self).__init__()
        self.conv1 = QConv(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, num_bit=num_bit,
                           wgt_sigma=wgt_sigma, wgt_temp=wgt_temp, act_sigma=act_sigma, act_temp=act_temp)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = QConv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, num_bit=num_bit,
                           wgt_sigma=wgt_sigma, wgt_temp=wgt_temp, act_sigma=act_sigma, act_temp=act_temp)
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

    def __init__(self, block, num_blocks, num_classes=10, num_bit=1, wgt_sigma=1, wgt_temp=2, act_sigma=2, act_temp=2):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, num_bit=num_bit,
                                       wgt_sigma=wgt_sigma, wgt_temp=wgt_temp, act_sigma=act_sigma, act_temp=act_temp)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, num_bit=num_bit,
                                       wgt_sigma=wgt_sigma, wgt_temp=wgt_temp, act_sigma=act_sigma, act_temp=act_temp)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, num_bit=num_bit,
                                       wgt_sigma=wgt_sigma, wgt_temp=wgt_temp, act_sigma=act_sigma, act_temp=act_temp)

        self.bn2 = nn.BatchNorm1d(64)

        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, num_bit, wgt_sigma, wgt_temp, act_sigma, act_temp):
        strides = [stride] + [1]*(num_blocks-1)
        ret_dict = dict()

        for i, stride in enumerate(strides):
            layers = []
            layers.append(block(self.in_planes, planes, stride, num_bit, wgt_sigma, wgt_temp, act_sigma, act_temp))
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


def resnet20_daq(**kwargs):
    """ResNet-20 model.
    """
    return ResNet(BasicBlock, [3, 3, 3], **kwargs)
