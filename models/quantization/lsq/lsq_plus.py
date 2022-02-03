import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from .lsq import Round
from .lsq import FunLSQ as WLSQPlus


class ALSQPlus(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, beta):
        ctx.save_for_backward(weight, alpha, beta)
        ctx.other = g, Qn, Qp
        w_q = Round.apply(torch.div((weight - beta), alpha)).clamp(Qn, Qp)
        w_q = w_q * alpha + beta
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha, beta = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = (weight - beta) / alpha
        smaller = (q_w < Qn).float()
        bigger = (q_w > Qp).float()
        between = 1.0 - smaller -bigger
        grad_alpha = ((smaller * Qn + bigger * Qp +
            between * Round.apply(q_w) - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0)
        grad_beta = ((smaller + bigger) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = between * grad_weight
        return grad_weight, grad_alpha,  None, None, None, grad_beta


class LSQPlusActQuantizer(nn.Module):
    def __init__(self, bits, qtype="quint", quant=False, observer=False, learning=False):
        super(LSQPlusActQuantizer, self).__init__()
        self.bits = bits
        self.qtype = qtype
        self.quant = quant
        self.observer = observer
        self.learning = learning
        assert self.bits != 1, "LSQ don't support binary quantization"
        assert self.qtype in ("qint", "quint"), "qtype just support qint or quint"
        if self.qtype == "quint":
            self.Qn = 0
            self.Qp = 2 ** self.bits - 1
        else:
            self.Qn = - 2 ** (self.bits - 1)
            self.Qp = 2 ** (self.bits - 1) - 1
        self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.zero_point = torch.nn.Parameter(torch.ones(0), requires_grad=True)
        self.grad_factor = 1.0
        self.observer_init = torch.tensor(1, dtype=torch.int8)

    def forward(self, x):
        if not self.quant:
            return x
        self.grad_factor = 1.0 / math.sqrt(x.numel() * self.Qp)
        if self.observer:
            min_val = torch.min(x.detach())
            if self.observer_init == 1:
                self.scale.data = (torch.max(x.detach()) - min_val) / (self.Qp - self.Qn)
                self.zero_point.data = min_val - self.scale.data * self.Qn
                self.observer_init = 0
            else:
                self.scale.data = self.scale.data * 0.9 + 0.1 * (torch.max(x.detach()) - min_val) / (self.Qp - self.Qn)
                self.zero_point.data = self.zero_point.data * 0.9 + 0.1 * (min_val - self.scale.data * self.Qn)

        if self.observer or self.learning:
            x = ALSQPlus.apply(x, self.scale, self.grad_factor, self.Qn, self.Qp, self.zero_point)
        return x


class LSQPlusWeightQuantizer(nn.Module):
    def __init__(self, bits, qtype="qint", per_channel=False, quant=False, observer=False, learning=False):
        super(LSQPlusWeightQuantizer, self).__init__()
        self.bits = bits
        self.qtype = qtype
        self.quant = quant
        self.observer = observer
        self.learning = learning
        assert self.bits != 1, "LSQ don't support binary quantization"
        assert self.qtype in ("qint", "quint"), "qtype just support qint or quint"
        if self.qtype == "quint":
            self.Qn = 0
            self.Qp = 2 ** bits - 1
        else:
            self.Qn = - 2 ** (bits - 1)
            self.Qp = 2 ** (bits - 1) - 1
        self.per_channel = per_channel
        self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.grad_factor = 1.0
        self.observer_init = torch.tensor(1, dtype=torch.int8)
        self.div = 2 ** self.bits - 1

    def forward(self, x):
        if not self.quant:
            return x
        self.grad_factor = 1.0 / math.sqrt(x.numel() * self.Qp)
        if self.observer:
            if self.per_channel:
                x_tmp = x.detach().contiguous().view(x.size()[0], -1)
                mean = torch.mean(x_tmp, dim=1)
                std = torch.std(x_tmp, dim=1)
                self.scale.data, _ = torch.max(torch.stack([torch.abs(mean - 3 * std), torch.abs(mean + 3 * std)]), dim=0)
                if self.observer_init == 1:
                    self.scale.data = self.scale.data / self.div
                    self.observer_init = 0
                else:
                    self.scale.data = self.scale.data * 0.9 + 0.1 * self.scale.data / self.div
            else:
                mean = torch.mean(x.detach())
                std = torch.std(x.detach())
                if self.observer_init == 1:
                    self.scale.data = max([torch.abs(mean - 3 * std), torch.abs(mean + 3 * std)]) / self.div
                    self.observer_init = 0
                else:
                    self.scale.data = self.scale.data*0.9 + \
                                      0.1 * max([torch.abs(mean - 3 * std), torch.abs(mean + 3 * std)]) / self.div
        if self.observer or self.learning:
            x = WLSQPlus.apply(x, self.scale, self.grad_factor, self.Qn, self.Qp, self.per_channel)
        return x


class LSQPlusConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 bits=8, quant_wgt=True, wgt_qtype="qint", wgt_per_channel=False, quant_act=True, act_qtype="quint",
                 quant=True, observer=False, learning=False, observer_step=1):
        super(LSQPlusConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.quant = quant
        self.quant_wgt = quant_wgt
        self.quant_act = quant_act
        self.act_quantizer = LSQPlusActQuantizer(bits, qtype=act_qtype,
                                                 quant=quant, observer=observer, learning=learning)
        self.weight_quantizer = LSQPlusWeightQuantizer(bits, qtype=wgt_qtype, per_channel=wgt_per_channel,
                                                       quant=quant, observer=observer, learning=learning)
        self.step = 0
        self.observer_step = observer_step

    def set_quant_config(self):
        if self.step <= self.observer_step:
            self.act_quantizer.quant = True
            self.act_quantizer.observer = True
            self.act_quantizer.learning = False

            self.weight_quantizer.quant = True
            self.weight_quantizer.observer = True
            self.weight_quantizer.learning = False
        else:
            self.act_quantizer.quant = True
            self.act_quantizer.observer = False
            self.act_quantizer.learning = True

            self.weight_quantizer.quant = True
            self.weight_quantizer.observer = False
            self.weight_quantizer.learning = True

    def forward(self, x):
        self.step += 1
        self.set_quant_config()
        if self.quant and self.quant_act:
            act = self.act_quantizer(x)
        else:
            act = x
        if self.quant and self.quant_wgt:
            wgt = self.weight_quantizer(self.weight)
        else:
            wgt = self.weight

        output = F.conv2d(act, wgt, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output
