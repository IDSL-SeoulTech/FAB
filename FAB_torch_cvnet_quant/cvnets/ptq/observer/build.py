#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
from .ema import EmaObserver
from .minmax import MinmaxObserver
from .olc import OlcObserver
from .omse import OmseObserver
from .percentile import PercentileObserver
from .ptf import PtfObserver

str2observer = {
    'minmax': MinmaxObserver,
    'ema': EmaObserver,
    'omse': OmseObserver,
    'percentile': PercentileObserver,
    'ptf': PtfObserver,
    'olc': OlcObserver
}


def build_observer(observer_str, module_type, bit_type, calibration_mode):
    observer = str2observer[observer_str]
    return observer(module_type, bit_type, calibration_mode)
