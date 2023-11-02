import torch.nn as nn
import torch as th
import torchvision.transforms as transforms
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from typing import Tuple, Union, List
from model.utils.nn_utils import Gaus2D, LambdaModule, TanhAlpha

class InitialLatentStates(nn.Module):
    def __init__(
            self, 
            gestalt_size: int, 
            num_objects: int, 
            bottleneck: str,
            size: Tuple[int, int],
            teacher_forcing: int
        ):
        super(InitialLatentStates, self).__init__()
        self.bottleneck = bottleneck

        self.num_objects                = num_objects
        self.gestalt_mean               = nn.Parameter(th.zeros(1, gestalt_size))
        self.gestalt_std                = nn.Parameter(th.ones(1, gestalt_size))
        self.std                        = nn.Parameter(th.zeros(1))
        self.gestalt_strength           = 2
        self.teacher_forcing            = teacher_forcing

        self.init = TanhAlpha(start = -1)
        self.register_buffer('priority', th.arange(num_objects).float() * 25, persistent=False)
        self.register_buffer('threshold', th.ones(1) * 0.8)
        self.last_mask = None
        self.binarize_first = round(gestalt_size * 0.8)

        self.gaus2d = nn.Sequential(
            Gaus2D((size[0] // 16, size[1] // 16)),
            Gaus2D((size[0] //  4, size[1] //  4)),
            Gaus2D(size)
        )

        self.level = 1
        self.t     = 0

        self.to_batch  = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o = num_objects))
        self.to_shared = LambdaModule(lambda x: rearrange(x, '(b o) c -> b (o c)', o = num_objects))

        self.blur = transforms.GaussianBlur(13)
        self.size = size

    def reset_state(self):
        self.last_mask = None
        self.t = 0
        self.to_next_spawn = 0

    def set_level(self, level):
        self.level = level
        factor = int(4 / (level ** 2))
        self.to_position = ErrorToPosition((self.size[0] //  factor, self.size[1] //  factor))

    def forward(
        self, 
        error: th.Tensor, 
        mask: th.Tensor = None, 
        position: th.Tensor = None,
        gestalt: th.Tensor = None,
        priority: th.Tensor = None,
        shuffleslots: bool = True, 
        slots_bounded_last: th.Tensor = None,
        slots_occlusionfactor_last: th.Tensor = None,
        allow_spawn: bool = True,
        clean_slots: bool = False
    ):

        batch_size = error.shape[0]
        device     = error.device

        if self.init.get() < 1:
            self.gestalt_strength = self.init()

        if self.last_mask is None:
            self.last_mask = th.zeros((batch_size * self.num_objects, 1), device = device)
            if shuffleslots:
                self.slots_assigned = th.ones((batch_size * self.num_objects, 1), device = device)
            else:
                self.slots_assigned = th.zeros((batch_size * self.num_objects, 1), device = device)

        if not allow_spawn:
            unnassigned = self.slots_assigned - slots_bounded_last
            self.slots_assigned = self.slots_assigned - unnassigned

        if clean_slots and (slots_occlusionfactor_last is not None):
            occluded = self.slots_assigned * (self.to_batch(slots_occlusionfactor_last) > 0.1).float()
            self.slots_assigned = self.slots_assigned - occluded

        if (slots_bounded_last is None) or (self.gestalt_strength < 1):

            if mask is not None:
                # maximum berechnung --> slot gebunden c=o
                mask2 = reduce(mask[:,:-1], 'b c h w -> (b c) 1' , 'max').detach()

                if self.gestalt_strength <= 0:
                    self.last_mask = mask2
                elif self.gestalt_strength < 1:
                    self.last_mask = th.maximum(self.last_mask, mask2)
                    self.last_mask = self.last_mask - th.relu(-1 * (mask2 - self.threshold) * (1 - self.gestalt_strength))
                else:
                    self.last_mask = th.maximum(self.last_mask, mask2)
    
            slots_bounded = (self.last_mask > self.threshold).float().detach() * self.slots_assigned
        else:
            slots_bounded = slots_bounded_last * self.slots_assigned

        if self.bottleneck == "binar":
            gestalt_new = repeat(th.sigmoid(self.gestalt_mean), '1 c -> b c', b = batch_size * self.num_objects)
            gestalt_new = gestalt_new + gestalt_new * (1 - gestalt_new) * th.randn_like(gestalt_new)
        else:
            gestalt_mean = repeat(self.gestalt_mean, '1 c -> b c', b = batch_size * self.num_objects)
            gestalt_std  = repeat(self.gestalt_std,  '1 c -> b c', b = batch_size * self.num_objects)
            gestalt_new  = th.sigmoid(gestalt_mean + gestalt_std * th.randn_like(gestalt_std))

        if gestalt is None:
            gestalt = gestalt_new
        else:
            gestalt = self.to_batch(gestalt) * slots_bounded + gestalt_new * (1 - slots_bounded)

        if priority is None:
            priority = repeat(self.priority, 'o -> (b o) 1', b = batch_size)
        else:
            priority = self.to_batch(priority) * slots_bounded + repeat(self.priority, 'o -> (b o) 1', b = batch_size) * (1 - slots_bounded)


        if shuffleslots:
            self.slots_assigned = th.ones_like(self.slots_assigned)

            xy_rand_new  = th.rand((batch_size * self.num_objects * 10, 2), device = device) * 2 - 1 
            std_new      = th.zeros((batch_size * self.num_objects * 10, 1), device = device)
            position_new = th.cat((xy_rand_new, std_new), dim=1) 

            position2d = self.gaus2d[self.level](position_new)
            position2d = rearrange(position2d, '(b o) 1 h w -> b o h w', b = batch_size)

            rand_error = reduce(position2d * error, 'b o h w -> (b o) 1', 'sum')

            xy_rand_new = rearrange(xy_rand_new, '(b r) c -> r b c', r = 10)
            rand_error  = rearrange(rand_error,  '(b r) c -> r b c', r = 10)

            max_error = th.argmax(rand_error, dim=0, keepdim=True)
            x, y = th.chunk(xy_rand_new, 2, dim=2)
            x = th.gather(x, dim=0, index=max_error).detach().squeeze(dim=0)
            y = th.gather(y, dim=0, index=max_error).detach().squeeze(dim=0)
            std  = repeat(self.std, '1 -> (b o) 1', b = batch_size, o=self.num_objects)

            if position is None:
                position = th.cat((x, y, std), dim=1) 
            else:
                position = self.to_batch(position) * slots_bounded + th.cat((x, y, std), dim=1) * (1 - slots_bounded)

        else:

            # set unassigned slots to empty position
            empty_position = th.tensor([-1,-1,0]).to(device)
            empty_position = repeat(empty_position, 'c -> (b o) c', b = batch_size, o=self.num_objects).detach()

            if position is None:
                position = empty_position
            else:
                position = self.to_batch(position) * self.slots_assigned + empty_position * (1 - self.slots_assigned)


            # blur errror, and set masked areas to zero
            error = self.blur(error)
            if mask is not None:
                mask2 = mask[:,:-1] * rearrange(slots_bounded, '(b o) 1 -> b o 1 1', b = batch_size)
                mask2 = th.sum(mask2, dim=1, keepdim=True)
                error = error * (1-mask2)
            max_error = reduce(error, 'b o h w -> (b o) 1', 'max')

            if self.to_next_spawn <= 0 and allow_spawn:

                self.to_next_spawn = 2

                # calculate the position with the highest error
                new_pos = self.to_position(error)
                std  = repeat(self.std, '1 -> b 1', b = batch_size)
                new_pos = repeat(th.cat((new_pos, std), dim=1), 'b c -> (b o) c', o = self.num_objects)
                
                #  calculate if an assigned slot is unbound (-->free)
                n_slots_assigned = self.to_shared(self.slots_assigned).sum(dim=1, keepdim=True)
                n_slots_bounded = self.to_shared(slots_bounded).sum(dim=1, keepdim=True)
                free_slot_given = th.clip(n_slots_assigned - n_slots_bounded, 0, 1)

                # either spawn a new slot or use the one that is free
                slots_new_index = n_slots_assigned * (1-free_slot_given) + n_slots_bounded * free_slot_given # reset the free slot each timespawn

                # new slot index
                free_slot_required = (max_error > 0).float()
                slots_new_index = F.one_hot(slots_new_index.long(), num_classes=self.num_objects+1).float().squeeze(dim=1)[:,:-1]
                slots_new_index = self.to_batch(slots_new_index * free_slot_required)

                # place new free slot
                position = new_pos * slots_new_index + position * (1 - slots_new_index)
                self.slots_assigned = th.clip(self.slots_assigned + slots_new_index, 0, 1)

        self.to_next_spawn -= 1
        return self.to_shared(position), self.to_shared(gestalt), self.to_shared(priority), error

    def get_slots_unassigned(self):
        return self.to_shared(1-self.slots_assigned)
    
    def get_slots_assigned(self):
        return self.to_shared(self.slots_assigned)
    

class OcclusionTracker(nn.Module):
    def __init__(self, batch_size, num_objects, device):
        super(OcclusionTracker, self).__init__()
        self.batch_size = batch_size
        self.num_objects = num_objects
        self.slots_bounded_all = th.zeros((batch_size * num_objects, 1)).to(device)
        self.threshold = 0.8
        self.device = device
        self.to_shared = LambdaModule(lambda x: rearrange(x, '(b o) c -> b (o c)', o = num_objects))
        self.slots_bounded_next_last = None

    def forward(
        self, 
        mask: th.Tensor = None, 
        rawmask: th.Tensor = None,
        reset_mask: bool = False,
        update: bool = True
    ):

        if mask is not None:

            # compute bounding mask
            slots_bounded_smooth_cur = reduce(mask[:,:-1], 'b o h w -> (b o) 1' , 'max').detach()
            slots_bounded_cur = (slots_bounded_smooth_cur > self.threshold).float().detach()
            if reset_mask:
                self.slots_bounded_next_last = slots_bounded_cur # allow immediate spawn
        
            if update:
                slots_bounded_cur = slots_bounded_cur * th.clip(self.slots_bounded_next_last + self.slots_bounded_all, 0, 1)
            else:
                self.slots_bounded_next_last = slots_bounded_cur
        
            if reset_mask:
                self.slots_bounded_smooth_all = slots_bounded_smooth_cur
                self.slots_bounded_all = slots_bounded_cur
            elif update:
                self.slots_bounded_all = th.maximum(self.slots_bounded_all, slots_bounded_cur)
                self.slots_bounded_smooth_all = th.maximum(self.slots_bounded_smooth_all, slots_bounded_smooth_cur)

            # compute occlusion mask
            slots_occluded_cur = self.slots_bounded_all - slots_bounded_cur

            # compute partially occluded mask
            mask = (mask[:,:-1] > self.threshold).float().detach()
            rawmask = (rawmask[:,:-1] > self.threshold).float().detach()
            masked = rawmask - mask

            masked = reduce(masked, 'b o h w -> (b o) 1' , 'sum')
            rawmask = reduce(rawmask, 'b o h w -> (b o) 1' , 'sum')

            slots_occlusionfactor_cur = (masked / (rawmask + 1)) * (1-slots_occluded_cur) + slots_occluded_cur
            slots_partially_occluded = (slots_occlusionfactor_cur > 0.1).float() #* slots_bounded_cur
            slots_fully_visible = (slots_occlusionfactor_cur <= 0.1).float() * slots_bounded_cur

            if reset_mask:
                self.slots_fully_visible_all = slots_fully_visible
            elif update:
                self.slots_fully_visible_all = th.maximum(self.slots_fully_visible_all, slots_fully_visible)

        return self.to_shared(self.slots_bounded_all), self.to_shared(self.slots_bounded_smooth_all), self.to_shared(slots_occluded_cur), self.to_shared(slots_partially_occluded), self.to_shared(slots_fully_visible), self.to_shared(slots_occlusionfactor_cur)

    def get_slots_fully_visible_all(self):
        return self.to_shared(self.slots_fully_visible_all)

class ErrorToPosition(nn.Module):
    def __init__(self, size: Union[int, Tuple[int, int]]):
        super(ErrorToPosition, self).__init__()

        self.register_buffer("grid_x", th.arange(size[0]), persistent=False)
        self.register_buffer("grid_y", th.arange(size[1]), persistent=False)

        self.grid_x = (self.grid_x / (size[0]-1)) * 2 - 1
        self.grid_y = (self.grid_y / (size[1]-1)) * 2 - 1

        self.grid_x = self.grid_x.view(1, 1, -1, 1).expand(1, 1, *size).clone()
        self.grid_y = self.grid_y.view(1, 1, 1, -1).expand(1, 1, *size).clone()

        self.grid_x = self.grid_x.view(1, 1, -1)
        self.grid_y = self.grid_y.view(1, 1, -1)

        self.size = size

    def forward(self, input: th.Tensor):
        assert input.shape[1] == 1

        input = rearrange(input, 'b c h w -> b c (h w)')
        argmax = th.argmax(input, dim=2, keepdim=True)

        x = self.grid_x[0,0,argmax].squeeze(dim=2)
        y = self.grid_y[0,0,argmax].squeeze(dim=2)

        return th.cat((x,y),dim=1)
    

def compute_rawmask(mask, bg_mask):

    num_objects = mask.shape[1]
    
    # d is a diagonal matrix which defines what to take the softmax over
    d_mask = th.diag(th.ones(num_objects+1)).to(mask.device)
    d_mask[:,-1] = 1
    d_mask[-1,-1] = 0

    # take subset of rawmask with the diagonal matrix
    rawmask = th.cat((mask, bg_mask), dim=1)
    rawmask = repeat(rawmask, 'b o h w -> b r o h w', r = num_objects+1)                 
    rawmask = rawmask[:,d_mask.bool()]
    rawmask = rearrange(rawmask, 'b (o r) h w -> b o r h w', o = num_objects)

    # take softmax between each object mask and the background mask
    rawmask = th.squeeze(th.softmax(rawmask, dim=2)[:,:,0], dim=2)
    rawmask = th.cat((rawmask, bg_mask), dim=1) # add background mask

    return rawmask