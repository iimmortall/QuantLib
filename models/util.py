#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from models.quantization.daq.daq import DAQConv2d
from models.quantization.lsq.lsq import LSQConv2d
from models.quantization.lsq.lsq_plus import LSQPlusConv2d


def get_func(func_name):
    func = globals().get(func_name)
    return func
