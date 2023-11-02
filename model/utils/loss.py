import torch as th
from torch import nn
from model.utils.nn_utils import SharedObjectsToBatch, LambdaModule
from einops import rearrange, repeat, reduce

__author__ = "Manuel Traub"
class PositionLoss(nn.Module):
    def __init__(self, num_objects: int):
        super(PositionLoss, self).__init__()

        self.to_batch  = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o = num_objects))

    def forward(self, position, position_last, slot_mask):
        
        slot_mask = rearrange(slot_mask, 'b o -> (b o) 1 1 1')
        position      = self.to_batch(position)
        position_last = self.to_batch(position_last).detach()

        return th.mean(slot_mask * (position - position_last)**2)

class ObjectModulator(nn.Module):
    def __init__(self, num_objects: int): 
        super(ObjectModulator, self).__init__()
        self.to_batch  = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o = num_objects))
        self.to_shared = LambdaModule(lambda x: rearrange(x, '(b o) c -> b (o c)', o = num_objects))
        self.position  = None
        self.gestalt   = None

    def reset_state(self):
        self.position = None
        self.gestalt  = None

    def forward(self, position: th.Tensor, gestalt: th.Tensor, slot_mask: th.Tensor):

        position = self.to_batch(position)
        gestalt  = self.to_batch(gestalt)
        slot_mask = self.to_batch(slot_mask)

        if self.position is None or self.gestalt is None:
            self.position = position.detach()
            self.gestalt  = gestalt.detach()
            return self.to_shared(position), self.to_shared(gestalt)

        _position = slot_mask * position + (1 - slot_mask) * self.position
        position  = th.cat((position[:,:-1], _position[:,-1:]), dim=1) # keep the position of the objects fixed
        gestalt   = slot_mask * gestalt  + (1 - slot_mask) * self.gestalt

        self.gestalt = gestalt.detach()
        self.position = position.detach()

        return self.to_shared(position), self.to_shared(gestalt)

class MoveToCenter(nn.Module):
    def __init__(self, num_objects: int):
        super(MoveToCenter, self).__init__()

        self.to_batch2d = SharedObjectsToBatch(num_objects)
        self.to_batch  = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o = num_objects))
    
    def forward(self, input: th.Tensor, position: th.Tensor):
        
        input    = self.to_batch2d(input) # b (o c) h w -> (b o) c h w
        position = self.to_batch(position).detach()
        position = th.stack((position[:,1], position[:,0]), dim=1)

        theta = th.tensor([1, 0, 0, 1], dtype=th.float, device=input.device).view(1,2,2)
        theta = repeat(theta, '1 a b -> n a b', n=input.shape[0])

        position = rearrange(position, 'b c -> b c 1')
        theta    = th.cat((theta, position), dim=2)

        grid   = nn.functional.affine_grid(theta, input.shape, align_corners=False)
        output = nn.functional.grid_sample(input, grid, align_corners=False)

        return output

class TranslationInvariantObjectLoss(nn.Module):
    def __init__(self, num_objects: int):
        super(TranslationInvariantObjectLoss, self).__init__()

        self.move_to_center  = MoveToCenter(num_objects)
        self.to_batch        = SharedObjectsToBatch(num_objects)
    
    def forward(
        self, 
        slot_mask: th.Tensor,
        object1: th.Tensor, 
        position1: th.Tensor,
        object2: th.Tensor, 
        position2: th.Tensor,
    ):
        slot_mask = rearrange(slot_mask, 'b o -> (b o) 1 1 1')
        object1 = self.move_to_center(th.sigmoid(object1 - 2.5), position1)
        object2 = self.move_to_center(th.sigmoid(object2 - 2.5), position2)

        return th.mean(slot_mask * (object1 - object2)**2)

