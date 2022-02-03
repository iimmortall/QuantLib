#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['DAQConv2d']


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


class DAQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 bits=1, quant_wgt=True, wgt_sigma=1, wgt_temp=2, quant_act=True, act_sigma=2, act_temp=2):
        super(DAQConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bits = bits
        self.quant_act = quant_act
        self.quant_wgt = quant_wgt
        self.bit_range = 2**self.bits - 1

        self.q_value = torch.from_numpy(np.linspace(0, 1, 2))
        self.q_value = self.q_value.reshape(len(self.q_value), 1, 1, 1, 1).float().cuda()
        self.wgt_sigma = wgt_sigma
        self.wgt_temp = wgt_temp
        self.act_sigma = act_sigma
        self.act_temp = act_temp

        self.register_buffer('init', torch.tensor(1).float().cuda())
        if self.quant_wgt:
            # using int32 max/min as init and back propagation to optimization
            # Weight
            self.uW = nn.Parameter(data=torch.tensor(2**31 - 1).float().cuda())
            self.lW = nn.Parameter(data=torch.tensor((-1) * (2**32)).float().cuda())
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

    def wgt_quant(self, x, u, l):
        # For reducing inference time
        x = self.clipping(x, u, l)
        delta = (u - l) / self.bit_range
        interval = (x - l) / delta
        x_floor = interval.floor()
        interval = interval - x_floor
        output = 2*(interval.round() + x_floor) - self.bit_range
        return output / self.bit_range

    def wgt_soft_quant(self, x, u, l):
        delta = (u - l) / self.bit_range
        interval = (x - l) / delta
        interval = torch.clamp(interval, min=0, max=self.bit_range)
        output = 2 * self.soft_argmax(interval, self.wgt_temp, self.wgt_sigma) - self.bit_range
        return output / self.bit_range

    def act_quant(self, x, u, l):
        # For reducing inference time
        x = self.clipping(x, u, l)
        delta = (u - l) / self.bit_range
        interval = (x - l) / delta
        x_floor = interval.floor()
        interval = interval - x_floor
        output = interval.round() + x_floor
        return output / self.bit_range

    def act_soft_quant(self, x, u, l):
        delta = (u - l) / self.bit_range
        interval = (x - l) / delta
        interval = torch.clamp(interval, min=0, max=self.bit_range)
        output = self.soft_argmax(interval, self.act_temp, self.act_sigma)
        return output / self.bit_range

    def soft_argmax(self, x, T, sigma):
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
        if not self.quant_act or not self.quant_wgt:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        if self.init:
            if self.quant_wgt:
                self.lW.data = torch.tensor(-3.0).cuda()
                self.uW.data = torch.tensor(3.0).cuda()
            if self.quant_act:
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
        if self.quant_wgt:
            if self.training:
                weight = self.wgt_soft_quant(norm_weight, curr_running_uw, curr_running_lw)
            else:
                weight = self.wgt_quant(norm_weight, curr_running_uw, curr_running_lw)
        else:
            weight = self.weight

        # Activation quantization
        if self.quant_act:
            if self.training:
                activation = self.act_soft_quant(x, curr_running_ua, curr_running_la)
            else:
                activation = self.act_quant(x, curr_running_ua, curr_running_la)
        else:
            activation = x

        if self.init == 1:
            # scale factor initialization
            q_output = F.conv2d(activation, weight, self.bias,  self.stride, self.padding, self.dilation, self.groups)
            ori_output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

            self.beta.data = torch.mean(torch.abs(ori_output)) / torch.mean(torch.abs(q_output))
            self.init = torch.tensor(0)

        output = F.conv2d(activation, weight, self.bias,  self.stride, self.padding, self.dilation, self.groups)
        output = torch.abs(self.beta) * output

        return output
