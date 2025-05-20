#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
from .log2 import Log2Quantizer
from .uniform import UniformQuantizer

str2quantizer = {'uniform': UniformQuantizer, 'log2': Log2Quantizer}


def build_quantizer(quantizer_str, bit_type, observer, module_type):
    quantizer = str2quantizer[quantizer_str]
    return quantizer(bit_type, observer, module_type)
