import torch.nn as nn
import torch as th
from model.nn.residual import SkipConnection, ResidualBlock
from model.utils.nn_utils import Gaus2D, SharedObjectsToBatch, BatchToSharedObjects, Prioritize
from einops import rearrange, repeat, reduce
from typing import Tuple, Union, List

__author__ = "Manuel Traub"

class PriorityEncoder(nn.Module):
    def __init__(self, num_objects, batch_size):
        super(PriorityEncoder, self).__init__()

        self.num_objects = num_objects
        self.register_buffer("indices", repeat(th.arange(num_objects), 'a -> b a', b=batch_size), persistent=False)

        self.index_factor    = nn.Parameter(th.ones(1))
        self.priority_factor = nn.Parameter(th.ones(1))

    def forward(self, priority: th.Tensor) -> th.Tensor:

        if priority is None:
            return None
        
        priority = priority * self.num_objects + th.randn_like(priority) * 0.1
        priority = priority * self.priority_factor 
        priority = priority + self.indices * self.index_factor

        return priority * 25


class GestaltPositionMerge(nn.Module):
    def __init__(
        self, 
        latent_size: Union[int, Tuple[int, int]],
        num_objects: int,
        batch_size: int
    ):

        super(GestaltPositionMerge, self).__init__()
        self.num_objects = num_objects

        self.gaus2d = Gaus2D(size=latent_size)

        self.to_batch  = SharedObjectsToBatch(num_objects)
        self.to_shared = BatchToSharedObjects(num_objects)

        self.prioritize = Prioritize(num_objects)

        self.priority_encoder = PriorityEncoder(num_objects, batch_size)

    def forward(self, position, gestalt, priority):
        
        position   = rearrange(position, 'b (o c) -> (b o) c', o = self.num_objects)
        gestalt    = rearrange(gestalt, 'b (o c) -> (b o) c 1 1', o = self.num_objects)
        priority   = self.priority_encoder(priority)

        position = self.gaus2d(position)
        position = self.to_batch(self.prioritize(self.to_shared(position), priority))

        return position * gestalt

class PatchUpscale(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor = 4, alpha = 1):
        super(PatchUpscale, self).__init__()
        assert in_channels % out_channels == 0
        
        self.skip = SkipConnection(in_channels, out_channels, scale_factor=scale_factor)

        self.residual = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels  = in_channels, 
                out_channels = in_channels, 
                kernel_size  = 3,
                padding      = 1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels  = in_channels, 
                out_channels = out_channels, 
                kernel_size  = scale_factor,
                stride       = scale_factor,
            ),
        )

        self.alpha = nn.Parameter(th.zeros(1) + alpha)

    def forward(self, input):
        return self.skip(input) + self.alpha * self.residual(input)


class LociDecoder(nn.Module):
    def __init__(
        self, 
        latent_size: Union[int, Tuple[int, int]],
        gestalt_size: int,
        num_objects: int, 
        img_channels: int,
        hidden_channels: int,
        level1_channels: int,
        num_layers: int,
        batch_size: int
    ): 

        super(LociDecoder, self).__init__()
        self.to_batch  = SharedObjectsToBatch(num_objects)
        self.to_shared = BatchToSharedObjects(num_objects)
        self.level     = 1

        assert(level1_channels % img_channels == 0)
        level1_factor   = level1_channels // img_channels
        print(f"Level1 channels: {level1_channels}")

        self.merge = GestaltPositionMerge(
            latent_size = latent_size,
            num_objects = num_objects,
            batch_size  = batch_size
        )

        self.layer0 = nn.Sequential(
            ResidualBlock(gestalt_size, hidden_channels, input_nonlinearity = False),
            *[ResidualBlock(hidden_channels, hidden_channels) for _ in range(num_layers-1)],
        )

        self.to_mask_level0 = ResidualBlock(hidden_channels, hidden_channels)
        self.to_mask_level1 = PatchUpscale(hidden_channels, 1)

        self.to_mask_level2 = nn.Sequential(
            ResidualBlock(hidden_channels, hidden_channels),
            ResidualBlock(hidden_channels, hidden_channels),
            PatchUpscale(hidden_channels, level1_factor, alpha = 1),
            PatchUpscale(level1_factor, 1, alpha = 1)
        )

        self.to_object_level0 = ResidualBlock(hidden_channels, hidden_channels)
        self.to_object_level1 = PatchUpscale(hidden_channels, img_channels)

        self.to_object_level2 = nn.Sequential(
            ResidualBlock(hidden_channels, hidden_channels),
            ResidualBlock(hidden_channels, hidden_channels),
            PatchUpscale(hidden_channels, level1_channels, alpha = 1),
            PatchUpscale(level1_channels, img_channels, alpha = 1)
        )

        self.mask_alpha   = nn.Parameter(th.zeros(1)+1e-16)
        self.object_alpha = nn.Parameter(th.zeros(1)+1e-16)


    def set_level(self, level):
        self.level = level

    def forward(self, position, gestalt, priority = None):

        maps    = self.layer0(self.merge(position, gestalt, priority))
        mask0   = self.to_mask_level0(maps)
        object0 = self.to_object_level0(maps)

        mask   = self.to_mask_level1(mask0)
        object = self.to_object_level1(object0)

        if self.level > 1:
            mask   = repeat(mask,   'b c h w -> b c (h h2) (w w2)', h2 = 4, w2 = 4)
            object = repeat(object, 'b c h w -> b c (h h2) (w w2)', h2 = 4, w2 = 4)

            mask   = mask   + self.to_mask_level2(mask0) * self.mask_alpha
            object = object + self.to_object_level2(object0) * self.object_alpha

        return self.to_shared(mask), self.to_shared(object)
