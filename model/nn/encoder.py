import torch.nn as nn
import torch as th
from model.utils.nn_utils import Gaus2D, BatchToSharedObjects, LambdaModule, ForcedAlpha, Binarize
from model.nn.residual import ResidualBlock, SkipConnection
from einops import rearrange, repeat, reduce
from typing import Tuple, Union, List

__author__ = "Manuel Traub"

class NeighbourChannels(nn.Module):
    def __init__(self, channels):
        super(NeighbourChannels, self).__init__()

        self.register_buffer("weights", th.ones(channels, channels, 1, 1), persistent=False)

        for i in range(channels):
            self.weights[i,i,0,0] = 0

    def forward(self, input: th.Tensor):
        return nn.functional.conv2d(input, self.weights)

class InputPreprocessing(nn.Module):
    def __init__(self, num_objects: int, size: Union[int, Tuple[int, int]]): 
        super(InputPreprocessing, self).__init__()
        self.num_objects = num_objects
        self.neighbours  = NeighbourChannels(num_objects)
        self.gaus2d      = Gaus2D(size)
        self.to_batch    = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o = num_objects))
        self.to_shared   = BatchToSharedObjects(num_objects)

    def forward(
        self, 
        input: th.Tensor, 
        error: th.Tensor, 
        mask: th.Tensor,
        object: th.Tensor,
        position: th.Tensor,
        rawmask: th.Tensor
    ):
        bg_mask     = repeat(mask[:,-1:], 'b 1 h w -> b c h w', c = self.num_objects)
        mask        = mask[:,:-1]
        mask_others = self.neighbours(mask)
        rawmask     = rawmask[:,:-1]

        own_gaus2d    = self.to_shared(self.gaus2d(self.to_batch(position)))
        
        input         = repeat(input,            'b c h w -> b o c h w', o = self.num_objects)
        error         = repeat(error,            'b 1 h w -> b o 1 h w', o = self.num_objects)
        bg_mask       = rearrange(bg_mask,       'b o h w -> b o 1 h w')
        mask_others   = rearrange(mask_others,   'b o h w -> b o 1 h w')
        mask          = rearrange(mask,          'b o h w -> b o 1 h w')
        object        = rearrange(object,        'b (o c) h w -> b o c h w', o = self.num_objects)
        own_gaus2d    = rearrange(own_gaus2d,    'b o h w -> b o 1 h w')
        rawmask       = rearrange(rawmask,       'b o h w -> b o 1 h w')
        
        output = th.cat((input, error, mask, mask_others, bg_mask, object, own_gaus2d, rawmask), dim=2) 
        output = rearrange(output, 'b o c h w -> (b o) c h w')

        return output

class PatchDownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 4, alpha = 1):
        super(PatchDownConv, self).__init__()
        assert out_channels % in_channels == 0
        
        self.layers = nn.Conv2d(
            in_channels  = in_channels, 
            out_channels = out_channels, 
            kernel_size  = kernel_size,
            stride       = kernel_size,
        )

        self.alpha = nn.Parameter(th.zeros(1) + alpha)
        self.kernel_size = 4
        self.channels_factor = out_channels // in_channels

    def forward(self, input: th.Tensor):
        k = self.kernel_size
        c = self.channels_factor
        skip = reduce(input, 'b c (h h2) (w w2) -> b c h w', 'mean', h2=k, w2=k)
        skip = repeat(skip, 'b c h w -> b (c n) h w', n=c)
        return skip + self.alpha * self.layers(input)

class AggressiveConvToGestalt(nn.Module):
    def __init__(self, channels, gestalt_size, size: Union[int, Tuple[int, int]]):
        super(AggressiveConvToGestalt, self).__init__()

        assert gestalt_size % channels == 0 or channels % gestalt_size == 0
        
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels  = channels, 
                out_channels = gestalt_size, 
                kernel_size  = 5,
                stride       = 3,
                padding      = 3
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels  = gestalt_size, 
                out_channels = gestalt_size, 
                kernel_size  = ((size[0] + 1)//3 + 1, (size[1] + 1)//3 + 1)
            )
        )
        if gestalt_size > channels:
            self.skip = nn.Sequential(
                LambdaModule(lambda x: reduce(x, 'b c h w -> b c 1 1', 'mean')),
                LambdaModule(lambda x: repeat(x, 'b c 1 1 -> b (c n) 1 1', n = gestalt_size // channels))
            )
        else:
            self.skip = LambdaModule(lambda x: reduce(x, 'b (c n) h w -> b c 1 1', 'mean', n = channels // gestalt_size))


    def forward(self, input: th.Tensor):
        return self.skip(input) + self.layers(input)

class PixelToPosition(nn.Module):
    def __init__(self, size: Union[int, Tuple[int, int]]):
        super(PixelToPosition, self).__init__()

        self.register_buffer("grid_x", th.arange(size[0]), persistent=False)
        self.register_buffer("grid_y", th.arange(size[1]), persistent=False)

        self.grid_x = (self.grid_x / (size[0]-1)) * 2 - 1
        self.grid_y = (self.grid_y / (size[1]-1)) * 2 - 1

        self.grid_x = self.grid_x.view(1, 1, -1, 1).expand(1, 1, *size).clone()
        self.grid_y = self.grid_y.view(1, 1, 1, -1).expand(1, 1, *size).clone()

        self.size = size

    def forward(self, input: th.Tensor):
        assert input.shape[1] == 1

        input = rearrange(input, 'b c h w -> b c (h w)')
        input = th.softmax(input, dim=2)
        input = rearrange(input, 'b c (h w) -> b c h w', h = self.size[0], w = self.size[1])

        x = th.sum(input * self.grid_x, dim=(2,3))
        y = th.sum(input * self.grid_y, dim=(2,3))

        return th.cat((x,y),dim=1)

class PixelToSTD(nn.Module):
    def __init__(self):
        super(PixelToSTD, self).__init__()
        self.alpha = ForcedAlpha()

    def forward(self, input: th.Tensor):
        assert input.shape[1] == 1
        return self.alpha(reduce(th.sigmoid(input - 10), 'b c h w -> b c', 'mean'))

class PixelToPriority(nn.Module):
    def __init__(self):
        super(PixelToPriority, self).__init__()

    def forward(self, input: th.Tensor):
        assert input.shape[1] == 1
        return reduce(th.tanh(input), 'b c h w -> b c', 'mean')

class LociEncoder(nn.Module):
    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]], 
        latent_size: Union[int, Tuple[int, int]],
        num_objects: int, 
        img_channels: int,
        hidden_channels: int,
        level1_channels: int,
        num_layers: int,
        gestalt_size: int,
        bottleneck: str
    ):
        super(LociEncoder, self).__init__()

        self.num_objects  = num_objects
        self.latent_size  = latent_size
        self.level        = 1

        self.to_shared = LambdaModule(lambda x: rearrange(x, '(b o) c -> b (o c)', o = self.num_objects))

        print(f"Level1 channels: {level1_channels}")

        self.preprocess = nn.ModuleList([
            InputPreprocessing(num_objects, (input_size[0] // 16, input_size[1] // 16)),
            InputPreprocessing(num_objects, (input_size[0] //  4, input_size[1] //  4)),
            InputPreprocessing(num_objects, (input_size[0], input_size[1]))
        ])

        self.to_channels = nn.ModuleList([
            SkipConnection(img_channels, hidden_channels),
            SkipConnection(img_channels, level1_channels),
            SkipConnection(img_channels, img_channels)
        ])

        self.layers2 = nn.Sequential(
            PatchDownConv(img_channels, level1_channels, alpha = 1e-16),
            *[ResidualBlock(level1_channels, level1_channels, alpha_residual=True) for _ in range(num_layers)]
        )

        self.layers1 = PatchDownConv(level1_channels, hidden_channels)

        self.layers0 = nn.Sequential(
            *[ResidualBlock(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )

        self.position_encoder = nn.Sequential(
            *[ResidualBlock(hidden_channels, hidden_channels) for _ in range(num_layers)],
            ResidualBlock(hidden_channels, 3),
        )

        self.xy_encoder = PixelToPosition(latent_size)
        self.std_encoder = PixelToSTD()
        self.priority_encoder = PixelToPriority()

        if bottleneck == "binar":
            print("Binary bottleneck")
            self.gestalt_encoder = nn.Sequential(
                *[ResidualBlock(hidden_channels, hidden_channels) for _ in range(num_layers)],
                AggressiveConvToGestalt(hidden_channels, gestalt_size, latent_size),
                LambdaModule(lambda x: rearrange(x, 'b c 1 1 -> b c')),
                Binarize(),
            )
            
        else:
            print("unrestricted bottleneck")
            self.gestalt_encoder = nn.Sequential(
                *[ResidualBlock(hidden_channels, hidden_channels) for _ in range(num_layers)],
                AggressiveConvToGestalt(hidden_channels, gestalt_size, latent_size),
                LambdaModule(lambda x: rearrange(x, 'b c 1 1 -> b c')),
                nn.Sigmoid(),
            )

    def set_level(self, level):
        self.level = level

    def forward(
        self, 
        input: th.Tensor,
        error: th.Tensor,
        mask: th.Tensor,
        object: th.Tensor,
        position: th.Tensor,
        rawmask: th.Tensor
    ):
        
        latent = self.preprocess[self.level](input, error, mask, object, position, rawmask)
        latent = self.to_channels[self.level](latent)

        if self.level >= 2:
            latent = self.layers2(latent)

        if self.level >= 1:
            latent = self.layers1(latent)

        latent  = self.layers0(latent)
        gestalt = self.gestalt_encoder(latent)

        latent   = self.position_encoder(latent)
        std      = self.std_encoder(latent[:,0:1])
        xy       = self.xy_encoder(latent[:,1:2])
        priority = self.priority_encoder(latent[:,2:3])

        position = self.to_shared(th.cat((xy, std), dim=1))
        gestalt  = self.to_shared(gestalt)
        priority = self.to_shared(priority)

        return position, gestalt, priority

