import torch.nn as nn
import torch as th
from model.nn.residual import ResidualBlock, SkipConnection, LinearResidual
from model.nn.encoder import PatchDownConv
from model.nn.encoder import AggressiveConvToGestalt
from model.nn.decoder import PatchUpscale
from model.utils.nn_utils import LambdaModule, Binarize
from einops import rearrange, repeat, reduce
from typing import Tuple

__author__ = "Manuel Traub"

class BackgroundEnhancer(nn.Module):
    def __init__(
        self, 
        input_size: Tuple[int, int], 
        batch_size,
    ):
        super(BackgroundEnhancer, self).__init__()

        self.batch_size = batch_size
        self.height = input_size[0]
        self.width  = input_size[1]
        self.mask   = nn.Parameter(th.ones(1, 1, *input_size) * 10)
        self.register_buffer('init', th.zeros(1).long())

    def get_init(self):
        return self.init.item()
    
    def forward(self, input: th.Tensor):
        
        mask = reduce(self.mask, '1 1 (h h2) (w w2) -> 1 1 h w', 'mean', h = input.shape[2], w = input.shape[3])
        mask = repeat(mask,      '1 1 h w -> b 1 h w', b = self.batch_size) * 0.1
        return mask
