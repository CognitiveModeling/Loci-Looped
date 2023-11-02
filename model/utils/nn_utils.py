from typing import Tuple
import torch.nn as nn
import torch as th
import numpy as np
from torch.autograd import Function
from einops import rearrange, repeat, reduce

class PushToInfFunction(Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.save_for_backward(tensor)
        return tensor.clone()

    @staticmethod
    def backward(ctx, grad_output):
        tensor = ctx.saved_tensors[0]
        grad_input = -th.ones_like(grad_output)
        return grad_input

class PushToInf(nn.Module):
    def __init__(self):
        super(PushToInf, self).__init__()
        
        self.fcn = PushToInfFunction.apply

    def forward(self, input: th.Tensor):
        return self.fcn(input)

class ForcedAlpha(nn.Module):
    def __init__(self, speed = 1):
        super(ForcedAlpha, self).__init__()

        self.init   = nn.Parameter(th.zeros(1))
        self.speed  = speed
        self.to_inf = PushToInf()

    def item(self):
        return th.tanh(self.to_inf(self.init * self.speed)).item()

    def forward(self, input: th.Tensor):
        return input * th.tanh(self.to_inf(self.init * self.speed))

class LinearInterpolation(nn.Module):
    def __init__(self, num_objects):
        super(LinearInterpolation, self).__init__()
        self.to_batch  = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o = num_objects))
        self.to_shared = LambdaModule(lambda x: rearrange(x, '(b o) c -> b (o c)', o = num_objects))

    def forward(
        self, 
        tensor_cur: th.Tensor = None,
        tensor_last: th.Tensor = None,
        slot_interpolation_value: th.Tensor = None
    ):

        slot_interpolation_value = rearrange(slot_interpolation_value, 'b o -> (b o) 1')
        tensor_cur = slot_interpolation_value * self.to_batch(tensor_last) + (1 - slot_interpolation_value) * self.to_batch(tensor_cur)

        return self.to_shared(tensor_cur)

class Gaus2D(nn.Module):
    def __init__(self, size: Tuple[int, int]):
        super(Gaus2D, self).__init__()

        self.size = size

        self.register_buffer("grid_x", th.arange(size[0]), persistent=False)
        self.register_buffer("grid_y", th.arange(size[1]), persistent=False)

        self.grid_x = (self.grid_x / (size[0]-1)) * 2 - 1
        self.grid_y = (self.grid_y / (size[1]-1)) * 2 - 1

        self.grid_x = self.grid_x.view(1, 1, -1, 1).expand(1, 1, *size).clone()
        self.grid_y = self.grid_y.view(1, 1, 1, -1).expand(1, 1, *size).clone()

    def forward(self, input: th.Tensor):

        x   = rearrange(input[:,0:1], 'b c -> b c 1 1')
        y   = rearrange(input[:,1:2], 'b c -> b c 1 1')
        std = rearrange(input[:,2:3], 'b c -> b c 1 1')

        x   = th.clip(x, -1, 1)
        y   = th.clip(y, -1, 1)
        std = th.clip(std, 0, 1)
            
        max_size = max(self.size)
        std_x = (1 + max_size * std) / self.size[0]
        std_y = (1 + max_size * std) / self.size[1]

        return th.exp(-1 * ((self.grid_x - x)**2/(2 * std_x**2) + (self.grid_y - y)**2/(2 * std_y**2)))

class Vector2D(nn.Module):
    def __init__(self, size: Tuple[int, int]):
        super(Vector2D, self).__init__()

        self.size = size

        self.register_buffer("grid_x", th.arange(size[0]), persistent=False)
        self.register_buffer("grid_y", th.arange(size[1]), persistent=False)

        self.grid_x = (self.grid_x / (size[0]-1)) * 2 - 1
        self.grid_y = (self.grid_y / (size[1]-1)) * 2 - 1

        self.grid_x = self.grid_x.view(1, 1, -1, 1).expand(1, 3, *size).clone()
        self.grid_y = self.grid_y.view(1, 1, 1, -1).expand(1, 3, *size).clone()

    def forward(self, input: th.Tensor, vector: th.Tensor = None):

        x   = rearrange(input[:,0:1], 'b c -> b c 1 1')
        y   = rearrange(input[:,1:2], 'b c -> b c 1 1')
        if vector is not None:
            x_vec = rearrange(vector[:,0:1], 'b c -> b c 1 1')
            y_vec = rearrange(vector[:,1:2], 'b c -> b c 1 1')

        x   = th.clip(x, -1, 1)
        y   = th.clip(y, -1, 1)
        std = 0.01
            
        max_size = max(self.size)
        std_x = (1 + max_size * std) / self.size[0]
        std_y = (1 + max_size * std) / self.size[1]
        grid = th.exp(-1 * ((self.grid_x - x)**2/(2 * std_x**2) + (self.grid_y - y)**2/(2 * std_y**2)))

        # interpolating between start and end point
        if vector is not None:
            for length in np.linspace(0, 1, 11):
                x_end = th.clip(x + x_vec * length, -1, 1)
                y_end = th.clip(y + y_vec * length, -1, 1)

                grid_point = th.exp(-1 * ((self.grid_x - x_end)**2/(0.5 * std_x**2) + (self.grid_y - y_end)**2/(0.5 * std_y**2)))
                grid_point[:, 0:2, :, :] = 0
                grid = th.max(grid, grid_point)

        return grid

class SharedObjectsToBatch(nn.Module):
    def __init__(self, num_objects):
        super(SharedObjectsToBatch, self).__init__()

        self.num_objects = num_objects

    def forward(self, input: th.Tensor):
        return rearrange(input, 'b (o c) h w -> (b o) c h w', o=self.num_objects)

class BatchToSharedObjects(nn.Module):
    def __init__(self, num_objects):
        super(BatchToSharedObjects, self).__init__()

        self.num_objects = num_objects

    def forward(self, input: th.Tensor):
        return rearrange(input, '(b o) c h w -> b (o c) h w', o=self.num_objects)

class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def forward(self, *x):
        return self.lambd(*x)

class PrintGradientFunction(Function):
    @staticmethod
    def forward(ctx, tensor, msg):
        ctx.msg = msg
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        print(f"{ctx.msg}: {th.mean(grad_output).item()} +- {th.std(grad_output).item()}")
        return grad_input, None

class PrintGradient(nn.Module):
    def __init__(self, msg = "PrintGradient"):
        super(PrintGradient, self).__init__()

        self.fcn = PrintGradientFunction.apply
        self.msg = msg

    def forward(self, input: th.Tensor):
        return self.fcn(input, self.msg)

class Prioritize(nn.Module):
    def __init__(self, num_objects):
        super(Prioritize, self).__init__()

        self.num_objects = num_objects
        self.to_batch    = SharedObjectsToBatch(num_objects)

    def forward(self, input: th.Tensor, priority: th.Tensor):
        
        if priority is None:
            return input

        batch_size = input.shape[0]
        weights    = th.zeros((batch_size, self.num_objects, self.num_objects, 1, 1), device=input.device)

        for o in range(self.num_objects):
            weights[:,o,:,0,0] = th.sigmoid(priority[:,:] - priority[:,o:o+1])
            weights[:,o,o,0,0] = weights[:,o,o,0,0] * 0

        input  = rearrange(input, 'b c h w -> 1 (b c) h w')
        weights = rearrange(weights, 'b o i 1 1 -> (b o) i 1 1')

        output = th.relu(input - nn.functional.conv2d(input, weights, groups=batch_size))
        output = rearrange(output, '1 (b c) h w -> b c h w ', b=batch_size)

        return output

class MultiArgSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super(MultiArgSequential, self).__init__(*args, **kwargs)

    def forward(self, *tensor):

        for n in range(len(self)):
            if isinstance(tensor, th.Tensor) or tensor == None:
                tensor = self[n](tensor)
            else:
                tensor = self[n](*tensor)

        return tensor

def create_grid(size):
    grid_x = th.arange(size[0])
    grid_y = th.arange(size[1])

    grid_x = (grid_x / (size[0]-1)) * 2 - 1
    grid_y = (grid_y / (size[1]-1)) * 2 - 1

    grid_x = grid_x.view(1, 1, -1, 1).expand(1, 1, *size).clone()
    grid_y = grid_y.view(1, 1, 1, -1).expand(1, 1, *size).clone()

    return th.cat((grid_y, grid_x), dim=1)

class Warp(nn.Module):
    def __init__(self, size, padding = 0.1):
        super(Warp, self).__init__()

        padding = int(max(size) * padding)
        padded_size = (size[0] + 2 * padding, size[1] + 2 * padding)

        self.register_buffer('grid', create_grid(size))
        self.register_buffer('padded_grid', create_grid(padded_size))

        self.replication_pad = nn.ReplicationPad2d(padding)
        self.interpolate = nn.Sequential(
            LambdaModule(lambda x:
                th.nn.functional.interpolate(x, size=size, mode='bicubic', align_corners = True)
            ),
            LambdaModule(lambda x: x - self.grid),
            nn.ConstantPad2d(padding, 0),
            LambdaModule(lambda x: x + self.padded_grid),
            LambdaModule(lambda x: rearrange(x, 'b c h w -> b h w c'))
        )

        self.warp = LambdaModule(lambda input, flow:
            th.nn.functional.grid_sample(input, flow, mode='bicubic', align_corners=True)
        )

        self.un_pad = LambdaModule(lambda x: x[:,:,padding:-padding,padding:-padding])
    
    def get_raw_flow(self, flow):
        return flow - self.grid

    def forward(self, input, flow):
        input = self.replication_pad(input)
        flow  = self.interpolate(flow)
        return self.un_pad(self.warp(input, flow))

class Binarize(nn.Module):
    def __init__(self):
        super(Binarize, self).__init__()

    def forward(self, input: th.Tensor):
        input = th.sigmoid(input)
        if not self.training:
            return th.round(input)

        return input + input * (1 - input) * th.randn_like(input)
    
class TanhAlpha(nn.Module):
    def __init__(self, start = 0, stepsize = 1e-4, max_value = 1):
        super(TanhAlpha, self).__init__()

        self.register_buffer('init', th.zeros(1) + start)
        self.stepsize  = stepsize
        self.max_value = max_value

    def get(self):
        return (th.tanh(self.init) * self.max_value).item()

    def forward(self):
        self.init = self.init.detach() + self.stepsize
        return self.get()