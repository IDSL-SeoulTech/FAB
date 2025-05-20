#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

from typing import Optional, Union

import torch
from torch import Tensor, nn

from cvnets.layers import ConvLayer2d
from cvnets.layers.activation import build_activation_layer
from cvnets.modules import BaseModule, SqueezeExcitation
from cvnets.ptq.bit_type import BIT_TYPE_DICT
from cvnets.ptq.layers import QAct, QConv2d, QOlc
from utils.math_utils import make_divisible


class InvertedResidualSE(BaseModule):
    """
    This class implements the inverted residual block with squeeze-excitation unit, as described in
    `MobileNetv3 <https://arxiv.org/abs/1905.02244>`_ paper

    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        dilation (Optional[int]): Use conv with dilation. Default: 1
        stride (Optional[int]): Use convolutions with a stride. Default: 1
        use_se (Optional[bool]): Use squeeze-excitation block. Default: False
        act_fn_name (Optional[str]): Activation function name. Default: relu
        se_scale_fn_name (Optional [str]): Scale activation function inside SE unit. Defaults to hard_sigmoid
        kernel_size (Optional[int]): Kernel size in depth-wise convolution. Defaults to 3.
        squeeze_factor (Optional[bool]): Squeezing factor in SE unit. Defaults to 4.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    """

    def __init__(
        self,
        opts,
        in_channels: int,
        out_channels: int,
        expand_ratio: Union[int, float],
        dilation: Optional[int] = 1,
        stride: Optional[int] = 1,
        use_se: Optional[bool] = False,
        act_fn_name: Optional[str] = "relu",
        se_scale_fn_name: Optional[str] = "hard_sigmoid",
        kernel_size: Optional[int] = 3,
        squeeze_factor: Optional[int] = 4,
        *args,
        **kwargs
    ) -> None:
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        act_fn = build_activation_layer(opts, act_type=act_fn_name, inplace=True)

        super().__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer2d(
                    opts,
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    use_act=False,
                    use_norm=True,
                ),
            )
            block.add_module(name="act_fn_1", module=act_fn)

        block.add_module(
            name="conv_3x3",
            module=ConvLayer2d(
                opts,
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=kernel_size,
                groups=hidden_dim,
                use_act=False,
                use_norm=True,
                dilation=dilation,
            ),
        )

        block.add_module(name="act_fn_2", module=act_fn)

        if use_se:
            se = SqueezeExcitation(
                opts=opts,
                in_channels=hidden_dim,
                squeeze_factor=squeeze_factor,
                scale_fn_name=se_scale_fn_name,
            )
            block.add_module(name="se", module=se)

        block.add_module(
            name="red_1x1",
            module=ConvLayer2d(
                opts,
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
                use_norm=True,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation
        self.use_se = use_se
        self.stride = stride
        self.act_fn_name = act_fn_name
        self.kernel_size = kernel_size
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        y = self.block(x)
        return x + y if self.use_res_connect else y

class InvertedResidual(BaseModule):
    """
    This class implements the inverted residual block, as described in `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper

    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (Optional[int]): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        dilation (Optional[int]): Use conv with dilation. Default: 1
        skip_connection (Optional[bool]): Use skip-connection. Default: True

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`

    """

    def __init__(
        self,
        opts,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        dilation: int = 1,
        skip_connection: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        self.quant = getattr(opts,"quant.quant")
        if(self.quant):
            calibration_c = getattr(opts,"quant.calibration_c")
            calibration_a = getattr(opts,"quant.calibration_a")
            calibration_w = getattr(opts,"quant.calibration_w")
            activation_bit = getattr(opts,"quant.activation_bit")
            act_bit_type = BIT_TYPE_DICT[activation_bit]

        block = nn.Sequential()

        if(self.quant):
            
            self.qact_shortcut = QAct(
            bit_type=act_bit_type,
            calibration_mode=calibration_a)

            self.conv1_1x1_olc_qact = QAct(
            bit_type=act_bit_type,
            calibration_mode=calibration_a)


        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer2d(
                    opts,
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    use_act=True,
                    use_norm=True,
                ),
            )
            if(self.quant):
                block.exp_1x1.block.conv = QConv2d(in_channels=block.exp_1x1.block.conv.in_channels,
                    out_channels=block.exp_1x1.block.conv.out_channels,
                    kernel_size=block.exp_1x1.block.conv.kernel_size,  
                    stride=block.exp_1x1.block.conv.stride, 
                    padding=block.exp_1x1.block.conv.padding,
                    dilation=block.exp_1x1.block.conv.dilation,  
                    groups=block.exp_1x1.block.conv.groups,
                    bias=block.exp_1x1.block.conv.bias,
                    calibration_mode = calibration_a
                )
                

        
        if(self.quant):
            self.conv2_3x3_olc = QOlc(in_chans=hidden_dim)
            self.qact1_1x1 = QAct(
            bit_type=act_bit_type,
            calibration_mode=calibration_a)
           

        block.add_module(
            name="conv_3x3",
            module=ConvLayer2d(
                opts,
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim,
                use_act=True,
                use_norm=True,
                dilation=dilation,
            ),
        )
        
        if(self.quant):
            block.conv_3x3.block.conv = QConv2d(in_channels=block.conv_3x3.block.conv.in_channels,
                out_channels=block.conv_3x3.block.conv.out_channels,
                kernel_size=block.conv_3x3.block.conv.kernel_size,  
                stride=block.conv_3x3.block.conv.stride, 
                padding=block.conv_3x3.block.conv.padding,
                dilation=block.conv_3x3.block.conv.dilation,  
                groups=block.conv_3x3.block.conv.groups,
                bias=block.conv_3x3.block.conv.bias,
                calibration_mode = calibration_c            # per-channel granularity for DWCL
            )
            block.conv_3x3.block.act = nn.ReLU6()

        if(self.quant):
            self.conv3_1x1_olc = QOlc(in_chans=hidden_dim)
            self.conv3_1x1_olc_qact = QAct(
                bit_type=act_bit_type,
                calibration_mode=calibration_a
            )

        block.add_module(
            name="red_1x1",
            module=ConvLayer2d(
                opts,
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
                use_norm=True,
            ),
        )
        if(self.quant):
            block.red_1x1.block.conv = QConv2d(in_channels=block.red_1x1.block.conv.in_channels,
                out_channels=block.red_1x1.block.conv.out_channels,
                kernel_size=block.red_1x1.block.conv.kernel_size,  
                stride=block.red_1x1.block.conv.stride, 
                padding=block.red_1x1.block.conv.padding,
                dilation=block.red_1x1.block.conv.dilation,  
                groups=block.red_1x1.block.conv.groups,
                bias=block.red_1x1.block.conv.bias,
                calibration_mode = calibration_a
            )
        
        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            if(self.quant):
                # shortcut
                shortcut = self.qact_shortcut(x)
                # EPL
                x = self.conv1_1x1_olc_qact(x)
                scale = self.conv1_1x1_olc_qact.quantizer.scale
                x = self.block.exp_1x1.block.conv(x,scale)
                x = self.block.exp_1x1.block.norm(x)
                x = self.block.exp_1x1.block.act(x)
            
                # DWCL
                x = self.qact1_1x1(x)   
                scale = self.qact1_1x1.quantizer.scale
                x = self.conv2_3x3_olc(x)
                x = self.block.conv_3x3.block.conv(x,scale)
                x = self.block.conv_3x3.block.norm(x)
                x = self.block.conv_3x3.block.act(x)
               
               
               
                # PJL
                x = self.conv3_1x1_olc_qact(x)
                scale = self.conv3_1x1_olc_qact.quantizer.scale
                x = self.block.red_1x1.block.conv(x,scale)
                x = self.block.red_1x1.block.norm(x)


                output = x + shortcut
                return output
            else:
                return x + self.block(x)
        else:
            if(self.quant):
                if(self.exp != 1):
                    # EPL
                    x = self.conv1_1x1_olc_qact(x)
                    scale = self.conv1_1x1_olc_qact.quantizer.scale
                    x = self.block.exp_1x1.block.conv(x,scale)
                    x = self.block.exp_1x1.block.norm(x)
                    x = self.block.exp_1x1.block.act(x)

                # DWCL
                x = self.qact1_1x1(x)
                scale = self.qact1_1x1.quantizer.scale 
                x = self.block.conv_3x3.block.conv(x,scale)
                x = self.block.conv_3x3.block.norm(x)
                x = self.block.conv_3x3.block.act(x)
              
                # PJL
                x = self.conv3_1x1_olc_qact(x)
                scale = self.conv3_1x1_olc_qact.quantizer.scale
                x = self.block.red_1x1.block.conv(x,scale)
                x = self.block.red_1x1.block.norm(x)

                return x

            else:
                return self.block(x)

