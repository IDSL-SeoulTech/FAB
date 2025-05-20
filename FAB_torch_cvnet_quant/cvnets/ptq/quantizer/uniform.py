#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.autograd import Function, Variable

from cvnets.ptq.observer.minmax import MinmaxObserver
from cvnets.ptq.observer.olc import OlcObserver

from .base import BaseQuantizer


class UniformQuantizer(BaseQuantizer):

    def __init__(self, bit_type, observer, module_type):
        super(UniformQuantizer, self).__init__(bit_type, observer, module_type)
        self.scale_factor = None
        self.zero = None
        self.scale_mask = None
        self.x_int_max = None
        self.scale = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.zero_point = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
      
    def update_quantization_params(self, *args, **kwargs):
        self.scale_factor, self.zero = self.observer.get_quantization_params(*args, **kwargs)

        self.scale.data = self.scale_factor
        self.zero_point.data = self.zero
        
    def quant(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)

        return QuantFunction.apply(inputs, scale, zero_point, self.bit_type.lower_bound, self.bit_type.upper_bound)
    
    def dequantize(self, inputs, scale=None, zero_point=None):
        
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        
        #outputs = (inputs - zero_point) * scale
        
        return DeQuantFunction.apply(inputs, scale, zero_point, self.bit_type.lower_bound, self.bit_type.upper_bound)
    
    def __repr__(self):
        #return f'UniformQuantizer(bit={self.bit_type.bits}, , sym={"Symmetric" if self.observer.symmetric else "Asymmetric"}, signed={self.bit_type.signed}, Gran={self.observer.calibration_mode}, type={self.module_type})'
        return f'UniformQuantizer(bit={self.bit_type.bits}, , signed={self.bit_type.signed}, Gran={self.observer.calibration_mode}, type={self.module_type})'


class QuantFunction(Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, lower_bound, upper_bound):
        #save for backward
        ctx.scale = scale
        ctx.zero_point = zero_point
        
        #quant
        x_q = x / scale + zero_point
        x_q = torch.round(x_q)
        x_q = torch.clamp(x_q,lower_bound,upper_bound)
        return x_q
    
    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        return grad_output.clone() / scale, None, None , None, None

class DeQuantFunction(Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, lower_bound, upper_bound):
        #save for backward
        ctx.scale = scale
        ctx.zero_point = zero_point
        x_dq = (x-zero_point) * scale
        return x_dq
    
    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        return grad_output.clone() * scale, None, None , None, None
    


