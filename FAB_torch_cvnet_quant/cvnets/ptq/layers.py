#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import decimal
import math
import time

import numpy as np

# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F

from .bit_type import BIT_TYPE_DICT
from .observer import build_observer
from .quantizer import build_quantizer


class QOlc(nn.Module):
    def __init__(self, in_chans, quant=False):
        super(QOlc, self).__init__()

        self.quant = quant
        self.mode = 'pass'
        self.in_chans = in_chans
        
        self.scale = torch.ones((1, self.in_chans, 1, 1)).to(device='cuda:0')
        self.olc_scale = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.add = torch.zeros((1, self.in_chans, 1, 1)).to(device='cuda:0')
        self.olc_add = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.ptf = None
        self.qkv_scale = torch.ones((3, 1, 1, 1, 1)).to(device='cuda:0')

    def forward(self,
                x,
                quantizer=None):

        self.olc_add.data = self.add
        self.olc_scale.data = self.scale
       
        # print(x.shape, len(x.shape))
        if self.mode == 'pass':
            if len(x.shape) == 4:
                x = (x - self.add) / self.scale
            elif len(x.shape) == 5: ## qkv
                x = x / self.qkv_scale
            else:
                self.scale = self.scale.view(1, 1, self.in_chans)
                x = (x - self.add) / self.scale

            return x

        elif self.mode == 'scaling':
            if len(x.shape) == 4:
                x = (x - self.add) / self.scale
            else:
                self.scale = self.scale.view(1, 1, self.in_chans)
                x = (x - self.add) / self.scale
            return x
        
class QuantWeightFunction(Function):
    @staticmethod
    def forward(ctx, w , scale, zero_point):
        new_quant_w = w / scale + zero_point
        new_quant_w = torch.round(new_quant_w)
        new_quant_w = new_quant_w.mul(scale) 
        ctx.scale = scale
        return new_quant_w
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None 
      
      
class QuantBiasFunction(Function):
    @staticmethod
    def forward(ctx, w , scale):
        new_quant_w = w / scale 
        int_quant_w = torch.round(new_quant_w)
        new_quant_w = int_quant_w.mul(scale) 
        return new_quant_w, int_quant_w
    
    @staticmethod
    def backward(ctx, grad_output, trash):
        return grad_output.clone(), None 
        

    
class round_ste(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
    
class clamp_ste(Function):
    @staticmethod
    def forward(ctx, x , lower_bound,upper_bound):
        return torch.clamp(x,lower_bound,upper_bound)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None
    

class floor_ste(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()




class QConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 quant=False,
                 qat = False,
                 calibrate=False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str
        self.bits = bit_type.bits
        self.module_type = 'conv_weight'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)
        # self.scale = nn.Parameter(torch.tensor(0.0),requires_grad=False)
        #self.bias_scale = nn.Parameter(torch.tensor(0.0),requires_grad=False)
        self.int_bias = nn.Parameter(torch.tensor(0.0),requires_grad=False)
        #self.int_weight = nn.Parameter(torch.tensor(0.0),requires_grad=False)
        
        self.qat = qat
        
    def bias_quantize(self, w, scale, n_bits=32): ## After BN Tuning
        w , q_bias= QuantBiasFunction.apply(w,scale)
        return w ,q_bias


    def forward(self, x, fused_scale_factor=None):
        if(fused_scale_factor is None):
            print("Test")
        if self.calibrate:
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)

        if not self.quant:
            return F.conv2d(x, self.weight, self.bias , self.stride, self.padding,
                        self.dilation, self.groups)
        
        if self.qat:
            with torch.no_grad():
                self.quantizer.observer.update(self.weight)
                self.quantizer.update_quantization_params(self.weight)
        #for fast training
        #self.int_weight.data = self.quantizer.quant(self.weight)
        weight = self.quantizer(self.weight)
        weight_scale , weight_zero =self.quantizer.scale, self.quantizer.zero_point

        # for fast training    
        # if(fused_scale_factor is not None):
        #     fused_scale_factor = fused_scale_factor * weight_scale
        #     self.bias_scale.data = fused_scale_factor

        # bias quantization
        if(self.bias is not None):
            if(fused_scale_factor is not None ):
                bias , int_bias = QuantBiasFunction.apply(self.bias,fused_scale_factor)
                self.bias.data = bias
                self.int_bias.data = int_bias

        # self.scale_factor.data , self.zero_point.data = self.quantizer.observer.get_quantization_params(self.weight)
    
        return F.conv2d(x, weight, self.bias , self.stride, self.padding,
                        self.dilation, self.groups)

             
    def __repr__(self):
        
        return f'Qconv2D (in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, groups={self.groups}, bit={self.bit_type.bits}, signed={self.bit_type.signed}, Gran={self.observer.calibration_mode})'
       


class QLinear(nn.Linear):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 quant=False,
                 calibrate=False,
                 qat = False,
                 last_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='minmax',
                 quantizer_str='uniform'):
        super(QLinear, self).__init__(in_features, out_features, bias)
        
        # self.q_weight = nn.Parameter(self.weight, requires_grad=False)
        # self.int_weight = nn.Parameter(self.weight, requires_grad=False)
        # self.scale_factor = nn.Parameter(torch.tensor(0.0),requires_grad=False)
        # self.zero_point = nn.Parameter(torch.tensor(0.0),requires_grad=False)
        self.bias_scale = nn.Parameter(torch.tensor(0.0),requires_grad=False)
        self.int_bias = nn.Parameter(torch.tensor(0.0),requires_grad=False)


        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = 'linear_weight'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)
        self.qat = qat
        
    def bias_quantize(self, w, n_bits=32,fused_scale_factor=None):
        w ,q_bias = QuantBiasFunction.apply(w,fused_scale_factor)
        return w, q_bias
        

    def forward(self, x, fused_scale_factor=None):
        if self.calibrate:
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)

        if not self.quant:
            return F.linear(x, self.weight, self.bias)
        
        if self.qat:
            with torch.no_grad():
                self.quantizer.observer.update(self.weight)
                self.quantizer.update_quantization_params(self.weight)

        #for fast training
        #self.int_weight.data =self.quantizer.quant(self.weight)

        weight = self.quantizer(self.weight)
        with torch.no_grad():
            # weight_scale, weight_zero = self.quantizer.observer.get_quantization_params()
            weight_scale, weight_zero = self.quantizer.scale , self.quantizer.zero_point

        # for fast training  
        if(fused_scale_factor is not None):
            with torch.no_grad():
                fused_scale_factor = fused_scale_factor * weight_scale
            self.bias_scale.data = fused_scale_factor
            
        # bias quantization
        if(self.bias is not None):
            if(fused_scale_factor is not None ):
                bias, int_bias = QuantBiasFunction.apply(self.bias,fused_scale_factor)
                self.bias.data = bias
                self.int_bias.data = int_bias
        #self.q_weight.data = weight
        #self.scale_factor.data , self.zero_point.data = self.quantizer.observer.get_quantization_params(self.weight)

        return F.linear(x, weight, self.bias)

def get_reshape_range(inputs):
        
    if len(inputs.shape) == 2:
        range_shape = (1, -1)
    elif len(inputs.shape) == 3:
        range_shape = (1, 1, -1)
    elif len(inputs.shape) == 4:
        range_shape = (1, -1, 1, 1)
    else:
        raise NotImplementedError
        
    return range_shape



class QAct(nn.Module):

    def __init__(self,
                 quant=False,
                 calibrate=False,
                 last_calibrate=False,
                 olc_calibrate=False,
                 bit_type=BIT_TYPE_DICT['int8'],
                 calibration_mode='layer_wise',
                 observer_str='percentile',
                 quantizer_str='uniform',
                 qat = False):
        super(QAct, self).__init__()

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.olc_calibrate = olc_calibrate
        self.bit_type = bit_type
        #self.bit_type = BIT_TYPE_DICT['int8']
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str
        self.qat = qat
        self.module_type = 'activation'
        self.observer = build_observer(self.observer_str, self.module_type,
                                       self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)
        self.olc_observer = build_observer("olc", self.module_type,
                                       self.bit_type, "layer_wise")
        self.olc_quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.olc_observer, self.module_type)
        
    def forward(self, x):        
        if self.calibrate:
            self.quantizer.observer.update(x)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
            return x
        if not self.quant:
            return x
        else:
            if self.qat:
                with torch.no_grad():
                    self.quantizer.observer.update(x)
                    self.quantizer.update_quantization_params(x)
            x = self.quantizer(x)
            return x
