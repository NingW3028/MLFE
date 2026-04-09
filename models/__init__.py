from os import path as osp
from typing import List, Text
import torch

__all__ = ['get_cell_based_tiny_net', 'CellStructure']

from .SharedUtils import change_key
from .cell_searchs import CellStructure
from .cell_infers import TinyNetwork


def get_cell_based_tiny_net(config):
    if config.name == 'infer.tiny':
        if hasattr(config, 'genotype'):
            genotype = config.genotype
        elif hasattr(config, 'arch_str'):
            genotype = CellStructure.str2structure(config.arch_str)
        else:
            raise ValueError('Can not find genotype from this config : {:}'.format(config))
        return TinyNetwork(config.C, config.N, genotype, config.num_classes)
    else:
        raise ValueError('invalid network name : {:}'.format(config.name))
