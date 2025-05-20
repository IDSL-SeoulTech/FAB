#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn

from .base import BaseObserver


class MinmaxObserver(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(MinmaxObserver, self).__init__(module_type, bit_type,
                                             calibration_mode)
        self.symmetric = self.bit_type.signed
        
    def update(self, v):
        with torch.no_grad():
            v = self.reshape_tensor(v)
            cur_max = v.max(axis=1).values
            if self.max_val is None:
                self.max_val = cur_max
            else:
                self.max_val = torch.max(cur_max, self.max_val)
            cur_min = v.min(axis=1).values
            if self.min_val is None:
                self.min_val = cur_min
            else:
                self.min_val = torch.min(cur_min, self.min_val)

            if self.calibration_mode == 'layer_wise':
                self.max_val = self.max_val.max()
                self.min_val = self.min_val.min()
            

    def get_quantization_params(self, *args, **kwargs):
        with torch.no_grad():
            max_val = self.max_val
            min_val = self.min_val
                    

            qmax = self.bit_type.upper_bound
            qmin = self.bit_type.lower_bound


            scale = torch.ones_like(max_val, dtype=torch.float32)
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)


            if self.symmetric:
                max_val = torch.max(-min_val, max_val)
                scale = max_val / (float(qmax - qmin) / 2)
                scale.clamp_(self.eps)
                zero_point = torch.zeros_like(max_val, dtype=torch.int64)
            else:
                scale = (max_val - min_val) / float(qmax - qmin)
                scale.clamp_(self.eps)
                zero_point = qmin - torch.round(min_val / scale)
                zero_point.clamp_(qmin, qmax)
                
        #self.s_factor = scale
        #self.z_point = zero_point
        return scale, zero_point
    
def float_to_fixed(input_tensor):
    return (input_tensor * (1 << 16)).int()

def fixed_to_float(input_tensor):
    return input_tensor.float() / (1 << 16)

def fixed(input):
    temp = float_to_fixed(input)
    result = fixed_to_float(temp)
    return result