# -*- coding: utf-8 -*-
import numpy as np
from os import path
from time2graph.utils.base_utils import Debugger
"""
    configuration file for benchmark datasets from UCR.
        Earthquakes (EQS).
        WormsTwoClass (WTC).
        StrawBerry (STB).
    including hyper-parameters and optimal arguments in xgboost.
"""

module_path = path.dirname(path.abspath(__file__))


ECG = {
    'K': 50,
    'C': 500,       # ou outro valor adequado para a quantidade de shapelet candidates
    'seg_length': 512,  # comprimento de cada ECG
    'num_segment': 8     # se o ECG não for segmentado
}

EQS = {
    'K': 50,
    'C': 800,
    'seg_length': 24,
    'num_segment': 21,
    'percentile': 5
}

WTC = {
    'K': 20,
    'C': 400,
    'seg_length': 30,
    'num_segment': 30,
    'percentile': 5,
    'global_flag': False
}

STB = {
    'K': 50,
    'C': 800,
    'seg_length': 15,
    'num_segment': 15,
    'percentile': 10,
    'embed': 'aggregate'
}

model_args = {
    'ecg': ECG,
    'ucr-Earthquakes': EQS,
    'ucr-WormsTwoClass': WTC,
    'ucr-Strawberry': STB
}

xgb_args = {
    'ucr-Earthquakes': {
        'max_depth': 16,
        'learning_rate': 0.2,
        'scale_pos_weight': 1,
        'booster': 'gbtree'
    },
    'ucr-WormsTwoClass': {
        'max_depth': 2,
        'learning_rate': 0.2,
        'scale_pos_weight': 1,
        'booster': 'gbtree'
    },
    'ucr-Strawberry': {
        'max_depth': 8,
        'learning_rate': 0.2,
        'scale_pos_weight': 1,
        'booster': 'gbtree'
    },
        'ecg': {
        'max_depth': 8,
        'learning_rate': 0.1,
        'scale_pos_weight': 6,  # ou 7, se a proporção for ligeiramente maior
        'n_estimators': 80,
        'booster': 'gbtree'
    }
}

__all__ = [
    'np',
    'path',
    'Debugger',
    'module_path',
    'model_args',
    'xgb_args'
]
