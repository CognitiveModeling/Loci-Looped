import torch as th
import torch.nn as nn
from einops import rearrange, repeat, reduce
from model.nn.percept_gate_controller import PerceptGateController
from model.nn.decoder import LociDecoder
from model.nn.encoder import LociEncoder
from model.nn.predictor import LociPredictor
from model.nn.background import BackgroundEnhancer
from model.utils.nn_utils import LinearInterpolation
from model.utils.loss import  ObjectModulator, TranslationInvariantObjectLoss, PositionLoss
from model.utils.slot_utils import OcclusionTracker, InitialLatentStates, compute_rawmask

class Loci(nn.Module):
    def __init__(
        self,
        cfg,
        teacher_forcing=1,
    ):
        super(Loci, self).__init__()

        self.teacher_forcing = teacher_forcing
        self.cfg = cfg

        self.encoder = LociEncoder(
            input_size       = cfg.input_size,
            latent_size      = cfg.latent_size,
            num_objects      = cfg.num_objects,
            img_channels     = cfg.img_channels * 2 + 6,
            hidden_channels  = cfg.encoder.channels,
            level1_channels  = cfg.encoder.level1_channels,
            num_layers       = cfg.encoder.num_layers,
            gestalt_size     = cfg.gestalt_size,
            bottleneck       = cfg.bottleneck,
        )

        self.percept_gate_controller = PerceptGateController(
            num_inputs  = 2*(cfg.gestalt_size + 3 + 1 + 1) + 3,
            num_hidden  = [32, 16],
            bias        = True,
            num_objects = cfg.num_objects,
            reg_lambda=cfg.update_module.reg_lambda,
        )

        self.predictor = LociPredictor(
            num_objects         = cfg.num_objects,
            gestalt_size        = cfg.gestalt_size,
            bottleneck          = cfg.bottleneck,
            channels_multiplier = cfg.predictor.channels_multiplier,
            heads               = cfg.predictor.heads,
            layers              = cfg.predictor.layers,
            reg_lambda          = cfg.predictor.reg_lambda,
            batch_size          = cfg.batch_size,
            transformer_type    = cfg.predictor.transformer_type,
        )

        self.decoder = LociDecoder(
            latent_size      = cfg.latent_size,
            num_objects      = cfg.num_objects,
            gestalt_size     = cfg.gestalt_size,
            img_channels     = cfg.img_channels,
            hidden_channels  = cfg.decoder.channels,
            level1_channels  = cfg.decoder.level1_channels,
            num_layers       = cfg.decoder.num_layers,
            batch_size       = cfg.batch_size,
        )
        
        self.background = BackgroundEnhancer(
            input_size        = cfg.input_size,
            gestalt_size      = cfg.background.gestalt_size,
            img_channels      = cfg.img_channels,
            depth             = cfg.background.num_layers, 
            latent_channels   = cfg.background.latent_channels,
            level1_channels   = cfg.background.level1_channels,
            batch_size        = cfg.batch_size,
        )

        self.initial_states = InitialLatentStates(
            gestalt_size               = cfg.gestalt_size,
            bottleneck                 = cfg.bottleneck,
            num_objects                = cfg.num_objects,
            size                       = cfg.input_size,
            teacher_forcing            = teacher_forcing
        )

        self.occlusion_tracker = OcclusionTracker(
            batch_size=cfg.batch_size,
            num_objects=cfg.num_objects,
            device=cfg.device
        )

        self.translation_invariant_object_loss = TranslationInvariantObjectLoss(cfg.num_objects)
        self.position_loss                     = PositionLoss(cfg.num_objects)
        self.modulator                         = ObjectModulator(cfg.num_objects)
        self.linear_gate                       = LinearInterpolation(cfg.num_objects)

        self.background.set_level(cfg.level)
        self.encoder.set_level(cfg.level)
        self.decoder.set_level(cfg.level)
        self.initial_states.set_level(cfg.level)

    def get_init_status(self):
        init = []
        for module in self.modules():
            if callable(getattr(module, "get_init", None)): 
                init.append(module.get_init())

        assert len(set(init)) == 1
        return init[0]

    def inc_init_level(self):
        for module in self.modules():
            if callable(getattr(module, "step_init", None)):
                module.step_init()

    def get_openings(self):
        return self.predictor.get_openings()

    def detach(self):
        for module in self.modules():
            if module != self and callable(getattr(module, "detach", None)):
                module.detach()

    def reset_state(self):
        for module in self.modules():
            if module != self and callable(getattr(module, "reset_state", None)):
                module.reset_state()

    def forward(self, *input, reset=True, detach=True, mode='end2end', evaluate=False, train_background=False, warmup=False, shuffleslots = True, reset_mask = False, allow_spawn = True, show_hidden = False, clean_slots = False):

        if detach:
            self.detach()

        if reset:
            self.reset_state()

        if train_background or self.get_init_status() < 1:
            return self.background(*input)

        return self.run_end2end(*input, evaluate=evaluate, warmup=warmup, shuffleslots = shuffleslots, reset_mask = reset_mask, allow_spawn = allow_spawn, show_hidden = show_hidden, clean_slots = clean_slots)

    def run_decoder(
        self, 
        position: th.Tensor, 
        gestalt: th.Tensor,
        priority: th.Tensor,
        bg_mask: th.Tensor,
        background: th.Tensor,
        only_mask: bool = False
    ):
        mask, object = self.decoder(position, gestalt, priority)

        rawmask = compute_rawmask(mask, bg_mask)
        mask   = th.softmax(th.cat((mask, bg_mask), dim=1), dim=1) 

        if only_mask:
            return mask, rawmask

        object = th.cat((th.sigmoid(object - 2.5), background), dim=1)
        _mask   = mask.unsqueeze(dim=2)
        _object = object.view(
            mask.shape[0], 
            self.cfg.num_objects + 1,
            self.cfg.img_channels,
            *mask.shape[2:]
        )

        output  = th.sum(_mask * _object, dim=1)
        return output, mask, object, rawmask
    
    def reset_unassigned_slots(self, position, gestalt, priority):
        
        position = self.linear_gate(position, th.zeros_like(position)-1, self.initial_states.get_slots_unassigned())
        gestalt  = self.linear_gate(gestalt,  th.zeros_like(gestalt),  self.initial_states.get_slots_unassigned())
        priority = self.linear_gate(priority, th.zeros_like(priority), self.initial_states.get_slots_unassigned())

        return  position, gestalt, priority

    def run_end2end(
        self, 
        input: th.Tensor,
        error_last: th.Tensor = None,
        mask_last: th.Tensor = None,
        rawmask_last: th.Tensor = None,
        position_last: th.Tensor = None,
        gestalt_last: th.Tensor = None,
        priority_last: th.Tensor = None,
        background = None,
        slots_occlusionfactor_last: th.Tensor = None,
        evaluate = False,
        warmup = False,
        shuffleslots = True,
        reset_mask = False,
        allow_spawn = True,
        show_hidden = False,
        clean_slots = False
    ):
        position_loss = th.tensor(0, device=input.device)
        time_loss     = th.tensor(0, device=input.device)
        bg_mask       = None
        position_encoder = None

        if error_last is None or mask_last is None:
            bg_mask = self.background(input, only_mask=True)
            error_last = th.sqrt(reduce((input - background)**2, 'b c h w -> b 1 h w', 'mean')).detach()

        position_last, gestalt_last, priority_last, error_cur = self.initial_states(
            error_last, mask_last, position_last, gestalt_last, priority_last, shuffleslots, self.occlusion_tracker.slots_bounded_all, slots_occlusionfactor_last, allow_spawn=allow_spawn, clean_slots=clean_slots
        )
        # only use assigned slots
        position_last, gestalt_last, priority_last = self.reset_unassigned_slots(position_last, gestalt_last, priority_last)

        if mask_last is None:
            mask_last, rawmask_last = self.run_decoder(position_last, gestalt_last, priority_last, bg_mask, background, only_mask=True)
        object_last_unprioritized = self.decoder(position_last, gestalt_last)[-1]

        # background and bg_mask for the next time point
        bg_mask = self.background(input, error_last, mask_last[:,-1:], only_mask=True)

        # position and gestalt for the current time point
        position_cur, gestalt_cur, priority_cur = self.encoder(input, error_last, mask_last, object_last_unprioritized, position_last, rawmask_last)
        if evaluate: 
            position_encoder = position_cur.clone().detach()

        # only use assigned slots
        position_cur, gestalt_cur, priority_cur = self.reset_unassigned_slots(position_cur, gestalt_cur, priority_cur)

        output_cur, mask_cur, object_cur, rawmask_cur = self.run_decoder(position_cur, gestalt_cur, priority_cur, bg_mask, background)
        slots_bounded,  slots_bounded_smooth, slots_occluded_cur, slots_partially_occluded_cur, slots_fully_visible_cur, slots_occlusionfactor_cur = self.occlusion_tracker(mask_cur, rawmask_cur, reset_mask)

        # do not project into the future in the warmup phase
        slots_closed = th.ones_like(repeat(slots_bounded, 'b o -> b o c', c=2))
        if warmup:
            position_next = position_cur
            gestalt_next  = gestalt_cur
            priority_next = priority_cur
        else:

            if self.cfg.inner_loop_enabled:

                # update module
                slots_closed =  (1-self.percept_gate_controller(position_cur, gestalt_cur, priority_cur, slots_occlusionfactor_cur, position_last, gestalt_last, priority_last, slots_occlusionfactor_last, self.position_last2, evaluate=evaluate))

                position_cur = self.linear_gate(position_cur, position_last, slots_closed[:, :, 1])
                priority_cur = self.linear_gate(priority_cur, priority_last, slots_closed[:, :, 1])
                gestalt_cur  = self.linear_gate(gestalt_cur,  gestalt_last,  slots_closed[:, :, 0])

            # position and gestalt for the next time point
            position_next, gestalt_next, priority_next = self.predictor(gestalt_cur, priority_cur, position_cur, slots_closed) 

        # combinded background and objects (masks) for next timepoint
        self.position_last2 = position_last.clone().detach()
        output_next, mask_next, object_next, rawmask_next = self.run_decoder(position_next, gestalt_next, priority_next, bg_mask, background)
        slots_bounded_next, slots_bounded_smooth_next, slots_occluded_next, slots_partially_occluded_next, slots_fully_visible_next, slots_occlusionfactor_next = self.occlusion_tracker(mask_next, rawmask_next, update=False)

        if evaluate:

            if show_hidden:
                pos_next = rearrange(position_next.clone(), '1 (o c) -> o c', c=3)
                largest_object = th.argmax(pos_next[:, 2], dim=0)
                pos_next[largest_object] = th.tensor([2, 2, 0.001])
                pos_next = rearrange(pos_next, 'o c -> 1 (o c)')
                output_hidden, _, object_hidden, rawmask_hidden = self.run_decoder(pos_next, gestalt_next, priority_next, bg_mask, background)
            else:
                output_hidden = None
                largest_object = None
                rawmask_hidden = None
                object_hidden = None

            return (
                output_next, 
                position_next, 
                gestalt_next, 
                priority_next,
                mask_next, 
                rawmask_next,
                object_next, 
                background, 
                slots_occlusionfactor_next,
                output_cur, 
                position_cur,
                gestalt_cur,
                priority_cur,
                mask_cur,
                rawmask_cur,
                object_cur,
                position_encoder,
                slots_bounded,
                slots_partially_occluded_cur,   
                slots_occluded_cur,
                slots_partially_occluded_next,
                slots_occluded_next,
                slots_closed,
                output_hidden,
                largest_object,
                rawmask_hidden,
                object_hidden
            )
            
        else:

            if not warmup:
                
                #regularize to small possition chananges over time
                position_loss = position_loss + self.position_loss(position_next, position_last.detach(), slots_bounded_smooth)

                # regularize to produce consistent object codes over time
                object_next_unprioritized = self.decoder(position_next, gestalt_next)[-1]
                time_loss  = time_loss + self.translation_invariant_object_loss(
                    slots_bounded_smooth,
                    object_last_unprioritized.detach(), 
                    position_last.detach(),
                    object_next_unprioritized,
                    position_next.detach(),
                )

            return (
                output_next, 
                output_cur, 
                position_next, 
                gestalt_next, 
                priority_next,
                mask_next, 
                rawmask_next,
                object_next, 
                background, 
                slots_occlusionfactor_next,
                position_loss,
                time_loss,
                slots_closed
            )

