#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import argparse
from typing import Optional

import torch
from torch import Tensor, nn

from cvnets.layers import (
    ConvLayer2d,
    Dropout,
    Identity,
    LinearLayer,
    LinearSelfAttention,
    MultiHeadAttention,
    SingleHeadAttention,
    StochasticDepth,
    get_normalization_layer,
)
from cvnets.layers.activation import build_activation_layer
from cvnets.modules import BaseModule
from cvnets.ptq.bit_type import BIT_TYPE_DICT
from cvnets.ptq.layers import QAct, QLinear
from utils import logger


class TransformerEncoder(BaseModule):
    """
    This class defines the pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_
    Args:
        opts: Command line arguments.
        embed_dim: :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`.
        ffn_latent_dim: Inner dimension of the FFN.
        num_heads: Number of heads in multi-head attention. Default: 8.
        attn_dropout: Dropout rate for attention in multi-head attention. Default: 0.0
        dropout: Dropout rate. Default: 0.0.
        ffn_dropout: Dropout between FFN layers. Default: 0.0.
        transformer_norm_layer: Normalization layer. Default: layer_norm.
        stochastic_dropout: Stochastic dropout setting. Default: 0.0.

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: Optional[int] = 8,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        transformer_norm_layer: Optional[str] = "batch_norm_1d",
        stochastic_dropout: Optional[float] = 0.0,
        *args,
        **kwargs,
    ) -> None:

        super().__init__()

        attn_unit = SingleHeadAttention(
            embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True
        )
        self.quant = getattr(opts,"quant.quant")
        if num_heads > 1:
            if(self.quant):
                calibration_a = getattr(opts,"quant.calibration_a")
                calibration_c = getattr(opts,"quant.calibration_c")
                act_bit_type = getattr(opts,"quant.activation_bit")
                act_bit = BIT_TYPE_DICT[act_bit_type]
                attn_unit = MultiHeadAttention(
                    embed_dim,
                    num_heads,
                    attn_dropout=attn_dropout,
                    bias=True,
                    coreml_compatible=getattr(
                        opts, "common.enable_coreml_compatible_module", False
                    ),
                    act_bit = act_bit,
                    quant = self.quant,
                )
            else:
                attn_unit = MultiHeadAttention(
                    embed_dim,
                    num_heads,
                    attn_dropout=attn_dropout,
                    bias=True,
                    coreml_compatible=getattr(
                        opts, "common.enable_coreml_compatible_module", False
                    ),
                )
                

        self.pre_norm_mha = nn.Sequential(
            get_normalization_layer(
                opts=opts, norm_type="batch_norm_1d", num_features=embed_dim
            ),
            attn_unit,
            Dropout(p=dropout),
        )

        act_name = build_activation_layer(opts, num_parameters=1)
        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(
                opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim
            ),
            #LinearLayer(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            QLinear(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            get_normalization_layer(
                opts=opts, norm_type=transformer_norm_layer, num_features=2*embed_dim
            ),
            act_name,
            Dropout(p=ffn_dropout),
            #LinearLayer(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            QLinear(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            get_normalization_layer(
                opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim
            ),
            Dropout(p=dropout),
        )
        
        self.pre_norm_ffn[3] = Lin_Swish()
       
        self.drop_path = Identity()
        if stochastic_dropout > 0.0:
            if dropout > 0.0:
                logger.error(
                    "Stochastic dropout and dropout are mutually exclusive. "
                    "Use either of them, but not both."
                    "Got: {} and {}".format(stochastic_dropout, dropout)
                )
            self.drop_path = StochasticDepth(p=stochastic_dropout, mode="row")

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.stochastic_dropout = stochastic_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__class__.__name__
        self.act_fn_name = act_name.__class__.__name__
        self.norm_type = transformer_norm_layer
        
        if(self.quant):
            self.qact_attn_out = QAct(bit_type=act_bit,
                                      calibration_mode=calibration_a)
            
            self.qact_ffn_1 = QAct(bit_type=act_bit)
            self.qact_1 = QAct(bit_type=act_bit)
            self.qact_2 = QAct(bit_type=act_bit)
            self.qact_3 = QAct(bit_type=act_bit)

    def __repr__(self) -> str:
        return "{}(embed_dim={}, ffn_dim={}, dropout={}, ffn_dropout={}, stochastic_dropout={}, attn_fn={}, act_fn={}, norm_fn={})".format(
            self.__class__.__name__,
            self.embed_dim,
            self.ffn_dim,
            self.std_dropout,
            self.ffn_dropout,
            self.stochastic_dropout,
            self.attn_fn_name,
            self.act_fn_name,
            self.norm_type,
        )

    def forward(
        self,
        x: Tensor,
        x_prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        last_scale_factor = None,
        *args,
        **kwargs,
    ) -> Tensor:

        # Multi-head attention
        res = x
        x = x.transpose(1,2)
        x = self.pre_norm_mha[0](x)  # norm
        x = x.transpose(1,2)
        x = self.pre_norm_mha[1](
            x_q=x,
            x_kv=x_prev,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            last_scale_factor=last_scale_factor,
            *args,
            **kwargs,
        )  # mha
        x = self.drop_path(self.pre_norm_mha[2](x))  # applying stochastic depth drop out
        x = x + res
        if(self.quant):
            x = self.qact_attn_out(x)
        # Feed forward network
        res = x
        ## norm block
        x = x.transpose(1,2)
        x = self.pre_norm_ffn[0](x) 
        x = x.transpose(1,2)
        ## lienar block
        scale = None
        if(self.quant):
            x = self.qact_ffn_1(x)
            with torch.no_grad():
                # scale , _ = self.qact_ffn_1.observer.get_quantization_params()
                scale = self.qact_ffn_1.quantizer.scale
        x = self.pre_norm_ffn[1](x,scale)  #linear 
        
        #norm block
        x = x.transpose(1,2)
        x = self.pre_norm_ffn[2](x) #norm
        x = x.transpose(1,2)
        x = self.pre_norm_ffn[3](x)
        ## swish block 
      
            
        if(self.quant):
            x = self.qact_1(x)  # after swish quant
            with torch.no_grad():
                # scale,_ = self.qact_1.observer.get_quantization_params()
                scale = self.qact_1.quantizer.scale
        x = self.pre_norm_ffn[4](x) #dropout

        # linear block
        x = self.pre_norm_ffn[5](x,scale) #linear
        #norm block
        x = x.transpose(1,2)
        x = self.pre_norm_ffn[6](x) #norm
        x = x.transpose(1,2)
        if(self.quant):
            x = self.qact_2(x)  # quant

        #dropoutblock
        x = self.pre_norm_ffn[7](x)

        #residual
        x = res + self.drop_path(x)
        scale = None
        if(self.quant):
            x = self.qact_3(x)
            # scale,_ = self.qact_3.observer.get_quantization_params()0
            scale = self.qact_3.quantizer.scale
        return x ,scale


class LinearAttnFFN(BaseModule):
    """
    This class defines the pre-norm transformer encoder with linear self-attention in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper
    Args:
        opts: command line arguments
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, P, N)`
        ffn_latent_dim (int): Inner dimension of the FFN
        attn_dropout (Optional[float]): Dropout rate for attention in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers. Default: 0.0
        norm_layer (Optional[str]): Normalization layer. Default: layer_norm_2d

    Shape:
        - Input: :math:`(B, C_{in}, P, N)` where :math:`B` is batch size, :math:`C_{in}` is input embedding dim,
            :math:`P` is number of pixels in a patch, and :math:`N` is number of patches,
        - Output: same shape as the input
    """

    def __init__(
        self,
        opts,
        embed_dim: int,
        ffn_latent_dim: int,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.1,
        ffn_dropout: Optional[float] = 0.0,
        norm_layer: Optional[str] = "layer_norm_2d",
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        attn_unit = LinearSelfAttention(
            opts, embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True
        )

        self.pre_norm_attn = nn.Sequential(
            get_normalization_layer(
                opts=opts, norm_type=norm_layer, num_features=embed_dim
            ),
            attn_unit,
            Dropout(p=dropout),
        )

        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(
                opts=opts, norm_type=norm_layer, num_features=embed_dim
            ),
            ConvLayer2d(
                opts=opts,
                in_channels=embed_dim,
                out_channels=ffn_latent_dim,
                kernel_size=1,
                stride=1,
                bias=True,
                use_norm=False,
                use_act=True,
            ),
            Dropout(p=ffn_dropout),
            ConvLayer2d(
                opts=opts,
                in_channels=ffn_latent_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                bias=True,
                use_norm=False,
                use_act=False,
            ),
            Dropout(p=dropout),
        )

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__repr__()
        self.norm_name = norm_layer

    def __repr__(self) -> str:
        return "{}(embed_dim={}, ffn_dim={}, dropout={}, ffn_dropout={}, attn_fn={}, norm_layer={})".format(
            self.__class__.__name__,
            self.embed_dim,
            self.ffn_dim,
            self.std_dropout,
            self.ffn_dropout,
            self.attn_fn_name,
            self.norm_name,
        )

    def forward(
        self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        if x_prev is None:
            # self-attention
            x = x + self.pre_norm_attn(x)
        else:
            # cross-attention
            res = x
            print(x.shape)
            x = self.pre_norm_attn[0](x)  # norm
            x = self.pre_norm_attn[1](x, x_prev)  # attn
            x = self.pre_norm_attn[2](x)  # drop
            x = x + res  # residual

        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x
