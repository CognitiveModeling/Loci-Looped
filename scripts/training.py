from copy import deepcopy
import torch as th
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
import cv2
import numpy as np
import os
from einops import rearrange, repeat, reduce
from scripts.utils.configuration import Configuration, Dict
from scripts.utils.io import WriterWrapper, init_device, model_path, LossLogger
from scripts.utils.optimizers import RAdam
from model.loci import Loci
import random
from scripts.utils.io import Timer
from scripts.utils.plot_utils import color_mask, plot_timestep
from scripts.validation import validation_bb, validation_clevrer, validation_adept
from scripts.exec.eval import main as eval_main

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_ DISABLE_SERVICE"] = "true"

def train_loci(cfg: Configuration, trainset: Dataset, valset: Dataset, file):

    # Set up cpu or gpu training
    device, verbose = init_device(cfg)
    if verbose:
        valset = Subset(valset, range(0, 8))
    use_wandb = (verbose == False)

    # generate random seed if not set
    if not ('seed' in cfg.defaults):
        cfg.defaults['seed'] = random.randint(0, 100)
    th.manual_seed(cfg.defaults.seed)

    # Define model path
    path = model_path(cfg, overwrite=False)
    cfg.save(path)
    os.makedirs(os.path.join(path, 'nets'), exist_ok=True)

    # Configure model
    cfg_net = cfg.model
    net = Loci(
        cfg                = cfg_net,
        teacher_forcing    = cfg.defaults.teacher_forcing
    )
    net = net.to(device=device)
    #net = th.compile(net)
    #net = th.jit.script(net) 

    # Log model size
    log_modelsize(net)
    writer = WriterWrapper(use_wandb, cfg)

    # Init Optimizers
    optimizer_init, optimizer_encoder, optimizer_decoder, optimizer_predictor, optimizer_background, optimizer_update = init_optimizer(cfg, net)
    scheduler = init_lr_scheduler_StepLR(cfg, optimizer_init, optimizer_encoder, optimizer_decoder, optimizer_predictor, optimizer_background, optimizer_update)

    # Option to load model
    if file != "":
        load_model(
            file,
            cfg,
            net,
            optimizer_init, 
            optimizer_encoder, 
            optimizer_decoder, 
            optimizer_predictor,
            optimizer_background,
            cfg.defaults.load_optimizers, 
            only_encoder_decoder = (cfg.num_updates == 0) # only load encoder and decoder for initial training
        )
        print(f'loaded {file}', flush=True)

    # Set up data loaders
    trainloader = get_loader(cfg, trainset, cfg_net, shuffle=True, verbose=verbose)
    valloader = get_loader(cfg, valset, cfg_net, shuffle=False, verbose=verbose)

    # initial save    
    save_model(
        os.path.join(path, 'nets', 'net0.pt'),
        net, 
        optimizer_init, 
        optimizer_encoder, 
        optimizer_decoder, 
        optimizer_predictor,
        optimizer_background
    )

    # Start training at num_updates
    num_updates = cfg.num_updates
    if num_updates > 0:
        print('!!! Start training at num_updates: ', num_updates)
        print('!!! Net init status: ', net.get_init_status())

    # Set up statistics
    loss_tracker = LossLogger(writer)

    # Init loss function
    imageloss = initialize_loss_function(cfg)

    # Set up training variables
    num_time_steps = 0
    bptt_steps = cfg.bptt.bptt_steps
    if not 'plot_interval' in cfg.defaults:
        cfg.defaults.plot_interval = 20000
    blackout_rate = cfg.blackout.blackout_rate if ('blackout' in cfg) else 0.0
    rollout_length = cfg.vp.rollout_length if ('vp' in cfg) else 0
    burnin_length = cfg.vp.burnin_length if ('vp' in cfg) else 0
    increase_bptt_steps = False
    background_blendin_factor = 0.0
    th.backends.cudnn.benchmark = True
    plot_next_sample = False
    timer = Timer()

    # Init net to current num_updates
    if num_updates >= cfg.phases.background_pretraining_end and net.get_init_status() < 1:
        net.inc_init_level()

    if num_updates >= cfg.phases.entity_pretraining_phase1_end and net.get_init_status() < 2:
        net.inc_init_level()

    if num_updates >= cfg.phases.entity_pretraining_phase2_end and net.get_init_status() < 3:
        net.inc_init_level()
        for param in optimizer_init.param_groups:
            param['lr'] = cfg.learning_rate.lr

    if num_updates > cfg.phases.start_inner_loop:
        net.cfg.inner_loop_enabled = True

    if num_updates >= cfg.phases.entity_pretraining_phase1_end:
        background_blendin_factor = max(min((num_updates - cfg.phases.entity_pretraining_phase1_end)/30000, 1.0), 0.0)


    # --- Start Training
    print('Start training')
    for epoch in range(cfg.max_epochs):

        # Validation every epoch
        if epoch >= 0:
            if cfg.datatype == 'adept':
                validation_adept(valloader, net, cfg, device, writer, epoch, path)
            elif cfg.datatype == 'clevrer':
                validation_clevrer(valloader, net, cfg, device, writer, epoch, path)
            elif cfg.datatype == "bouncingballs" and (epoch % 2 == 0):
                validation_bb(valloader, net, cfg, device, writer, epoch, path)

        # Start epoch training
        print('Start epoch:', epoch)

        # Backprop through time steps
        if increase_bptt_steps:
            bptt_steps = min(bptt_steps + 1, cfg.bptt.bptt_steps_max)
            print('Increase closed loop steps to', bptt_steps)
        increase_bptt_steps = False

        for batch_index, input in enumerate(trainloader):

            # Extract input and background 
            tensor          = input[0]
            background      = input[1].to(device)
            shuffleslots    = (num_updates <= cfg.phases.shufleslots_end)

            # Placeholders
            position    = None
            gestalt     = None
            priority    = None
            mask        = None
            object      = None
            rawmask     = None
            loss        = th.tensor(0)
            summed_loss = None
            slots_occlusionfactor = None

            # Apply skip frames to sequence
            start = random.randrange(cfg.defaults.skip_frames) if rollout_length == 0 else random.randrange(0, (tensor.shape[1]-(rollout_length+burnin_length)))
            selec = range(start, tensor.shape[1], cfg.defaults.skip_frames)
            tensor = tensor[:,selec]
            sequence_len = tensor.shape[1]

            # Initial frame
            input      = tensor[:,0].to(device)
            input_next = input
            target     = th.clip(input, 0, 1).detach()
            error_last = None

            # plotting mode
            plot_this_sample = plot_next_sample
            plot_next_sample = False
            video_list = []
            num_rollout = 0

            # First apply teacher forcing for the first x frames
            for t in range(-cfg.defaults.teacher_forcing, sequence_len-1):

                # Set update scheme for backprop through time 
                if t >= cfg.bptt.bptt_start_timestep:
                    t_run = (t - cfg.bptt.bptt_start_timestep)
                    run_optimizers = t_run % bptt_steps == bptt_steps - 1
                    detach         = (t_run % bptt_steps == 0) or t == -cfg.defaults.teacher_forcing
                else:
                    run_optimizers = True
                    detach         = True

                if verbose:
                    print(f't: {t}, run_optimizers: {run_optimizers}, detach: {detach}')

                if t >= 0:
                    # Skip to next frame
                    num_time_steps += 1
                    input      = input_next
                    input_next = tensor[:,t+1].to(device)
                    target     = th.clip(input_next, 0, 1)

                    # Apply error dropout
                    if net.get_init_status() > 2 and cfg.defaults.error_dropout > 0 and np.random.rand() < cfg.defaults.error_dropout:
                        error_last = th.zeros_like(error_last)

                    # Apply sensation blackout when training clevrer
                    if net.cfg.inner_loop_enabled and blackout_rate > 0:
                        if t >= cfg.blackout.blackout_start_timestep:
                            blackout    = th.tensor((np.random.rand(cfg_net.batch_size) < blackout_rate)[:,None,None,None]).float().to(device)
                            input       = blackout * (input * 0)         + (1-blackout) * input
                            error_last  = blackout * (error_last * 0)    + (1-blackout) * error_last

                    # Apply rollout training
                    if net.cfg.inner_loop_enabled and (rollout_length > 0) and (burnin_length-1) < t:
                            input       = input * 0
                            error_last  = error_last * 0
                            num_rollout += 1
                            
                            if (burnin_length + rollout_length -1) == t:
                                run_optimizers = True
                                detach         = True

                            if (burnin_length + rollout_length) == t:
                                break


                # Forward Pass
                (
                    output_next, 
                    output_cur,
                    position, 
                    gestalt, 
                    priority, 
                    mask, 
                    rawmask,
                    object, 
                    background, 
                    slots_occlusionfactor,
                    position_loss,
                    time_loss,
                    latent_loss,
                    slots_closed,
                    slots_bounded
                    ) = net(
                    input,      # current frame
                    error_last, # error of last frame --> missing object
                    mask,       # masks of current frame
                    rawmask,    # raw masks of current frame
                    position,   # positions of objects of next frame
                    gestalt,    # gestalt of objects of next frame
                    priority,   # priority of objects of next frame
                    background,
                    slots_occlusionfactor,
                    reset = (t == -cfg.defaults.teacher_forcing), # new sequence
                    warmup = (t < 0),                    # teacher forcing
                    detach = detach,
                    shuffleslots = shuffleslots or ((cfg.datatype == 'clevrer') and (t<=0)),
                    reset_mask = (t <= 0),
                    clean_slots = (t <= 0 and not shuffleslots),
                )

                # Loss weighting 
                position_loss = position_loss * cfg_net.position_regularizer
                time_loss     = time_loss     * cfg_net.time_regularizer
                if latent_loss.item() > 0:
                    latent_loss   = latent_loss   * cfg_net.latent_regularizer

                # Compute background error
                bg_error_cur  = th.sqrt(reduce((input - background)**2, 'b c h w -> b 1 h w', 'mean')).detach()
                bg_error_next = th.sqrt(reduce((target - background)**2, 'b c h w -> b 1 h w', 'mean')).detach()

                # Compute next-frame prediction error
                error_next    = th.sqrt(reduce((target - output_next)**2, 'b c h w -> b 1 h w', 'mean')).detach()
                error_next    = th.sqrt(error_next) * bg_error_next
                error_last    = error_next

                # Initially focus on foreground learning
                if background_blendin_factor < 1:
                    fg_mask_next = th.gt(bg_error_next, 0.1).float().detach()
                    fg_mask_next[fg_mask_next == 0] = background_blendin_factor
                    target       = th.clip(target * fg_mask_next, 0, 1)

                    fg_mask_cur = th.gt(bg_error_cur, 0.1).float().detach()
                    fg_mask_cur[fg_mask_cur == 0] = background_blendin_factor
                    input       = th.clip(input * fg_mask_cur, 0, 1)

                    # Gradually blend in background for more stable training
                    if num_updates % 30 == 0 and num_updates >= cfg.phases.entity_pretraining_phase1_end:
                        background_blendin_factor = min(1, background_blendin_factor + 0.001)

                # Final Loss computation
                encoding_loss = imageloss(output_cur, input) * cfg_net.encoder_regularizer
                prediction_loss = imageloss(output_next, target)
                loss = prediction_loss + encoding_loss + position_loss + time_loss + latent_loss
                
                # apply loss decay according to num_rollout
                if rollout_length > 0:
                    loss = loss * 0.75**(num_rollout-1)

                # Accumulate loss over BPP steps
                summed_loss = loss if summed_loss is None else summed_loss + loss
                mask = mask.detach()

                if run_optimizers:

                    # detach gradients for next step
                    position = position.detach()
                    gestalt  = gestalt.detach()
                    rawmask  = rawmask.detach()
                    object   = object.detach()
                    priority = priority.detach()

                    # zero grad
                    optimizer_init.zero_grad()
                    optimizer_encoder.zero_grad()
                    optimizer_decoder.zero_grad()
                    optimizer_predictor.zero_grad()
                    optimizer_background.zero_grad()
                    if net.cfg.inner_loop_enabled:
                        optimizer_update.zero_grad()

                    # optimize
                    summed_loss.backward()
                    optimizer_init.step()
                    optimizer_encoder.step()
                    optimizer_decoder.step()
                    optimizer_predictor.step()
                    optimizer_background.step()
                    if net.cfg.inner_loop_enabled:
                        optimizer_update.step()

                    # Reset loss
                    num_updates += 1
                    summed_loss = None

                    # Update net status
                    update_net_status(num_updates, net, cfg, optimizer_init)
                    step_lr_scheduler(scheduler)

                    if num_updates == cfg.phases.start_inner_loop:
                        print('Start inner loop')
                        net.cfg.inner_loop_enabled = True

                    if (cfg.bptt.increase_bptt_steps_every > 0) and ((num_updates-cfg.num_updates) % cfg.bptt.increase_bptt_steps_every == 0) and ((num_updates-cfg.num_updates) > 0):
                        increase_bptt_steps = True

                    if net.cfg.inner_loop_enabled and ('blackout' in cfg) and (cfg.blackout.blackout_increase_every > 0) and ((num_updates-cfg.num_updates) % cfg.blackout.blackout_increase_every == 0) and ((num_updates-cfg.num_updates) > 0):
                        blackout_rate = min(blackout_rate + cfg.blackout.blackout_increase_rate, cfg.blackout.blackout_rate_max)

                    # Plots for online evaluation
                    if num_updates % cfg.defaults.plot_interval == 0:
                        plot_next_sample = True

                if plot_this_sample:
                    img_tensor = plot_online(cfg, path, f'{epoch}_{batch_index}', input, background, mask, sequence_len, t, output_next, bg_error_next)
                    video_list.append(img_tensor)

                # Track statisitcs
                if t >= cfg.defaults.statistics_offset:
                    track_statistics(cfg, net, loss_tracker, input, gestalt, mask, target, output_next, output_cur, encoding_loss, prediction_loss, position_loss, time_loss, latent_loss, slots_closed, slots_bounded, bg_error_cur, bg_error_next, scheduler, num_updates)
                    loss_tracker.update_average_loss(loss.item(), num_updates)
                    writer.add_scalar('Train/BPTT_steps', bptt_steps, num_updates)
                    writer.add_scalar('Train/Background_Blendin', background_blendin_factor, num_updates)
                    writer.add_scalar('Train/Blackout_Rate', blackout_rate, num_updates)
    
                # Logging
                if num_updates % 100 == 0 and run_optimizers:
                    print(f'Epoch[{num_updates}/{num_time_steps}/{sequence_len}]: {str(timer)}, {epoch + 1}, Blendin:{float(background_blendin_factor)}, i: {net.get_init_status() + net.initial_states.init.get():.2f},' + loss_tracker.get_log(), flush=True)

                # Training finished
                net_path = None
                if num_updates > cfg.max_updates:
                    net_path = os.path.join(path, 'nets', 'net_final.pt')
                if num_updates % 50000 == 0 and run_optimizers:
                    net_path = os.path.join(path, 'nets', f'net_{num_updates}.pt')

                if net_path is not None:
                    save_model(
                        net_path,
                        net, 
                        optimizer_init, 
                        optimizer_encoder, 
                        optimizer_decoder, 
                        optimizer_predictor,
                        optimizer_background
                    )
                    if ('final' in net_path) or ('num_objects_test' in cfg.model):
                        eval_main(net_path, 1, deepcopy(cfg))

                    if 'final' in net_path:
                        print("Training finished")
                        writer.flush()
                        return

            if plot_this_sample:
                video_tensor = rearrange(th.stack(video_list, dim=0)[None], 'b t h w c -> b t c h w')
                writer.add_video('Train/Video', video_tensor, num_updates)

    pass

def track_statistics(cfg, net, loss_tracker, input, gestalt, mask, target, output_next, output_cur, encoding_loss, prediction_loss, position_loss, time_loss, latent_loss, slots_closed, slots_bounded, bg_error_cur, bg_error_next, scheduler, num_updates):
    
    # area of foreground mask
    num_objects = th.mean(reduce((reduce(mask[:,:-1], 'b c h w -> b c', 'max') > 0.5).float(), 'b c -> b', 'sum')).item()

    # difference in shape
    _gestalt  = reduce(th.min(th.abs(gestalt), th.abs(1 - gestalt)),    'b (o c) -> (b o)', 'mean', o = cfg.model.num_objects)
    _gestalt2 = reduce(th.min(th.abs(gestalt), th.abs(1 - gestalt))**2, 'b (o c) -> (b o)', 'mean', o = cfg.model.num_objects)
    max_mask     = (reduce(mask[:,:-1], 'b c h w -> (b c)', 'max') > 0.5).float()
    avg_gestalt = (th.sum(_gestalt  * max_mask) / (1e-16 + th.sum(max_mask)))
    avg_gestalt2 = (th.sum(_gestalt2 * max_mask) / (1e-16 + th.sum(max_mask)))
    avg_gestalt_mean = th.mean(th.clip(gestalt, 0, 1))

    # udpdate gates
    num_bounded = reduce(slots_bounded, 'b o -> b', 'sum').mean().item()
    slots_closed = slots_closed * slots_bounded[:,:,None]
    avg_update_gestalt = slots_closed[:,:,0].sum()/slots_bounded.sum() if slots_bounded.sum() > 0 else 0.0
    avg_update_position = slots_closed[:,:,1].sum()/slots_bounded.sum() if slots_bounded.sum() > 0 else 0.0
    avg_update_gestalt = float(avg_update_gestalt)
    avg_update_position = float(avg_update_position)

    # Prediction Loss + Encoder Loss as MSE + only foreground pixels
    mseloss = nn.MSELoss()
    loss_next = mseloss(output_next * bg_error_next, target * bg_error_next)
    loss_cur  = mseloss(output_cur * bg_error_cur,   input  * bg_error_cur)

    # learning rate
    if scheduler is not None:
        lr = scheduler[0].get_last_lr()[0]
    else:
        lr = cfg.learning_rate.lr

    # gatelORD openings
    openings = th.mean(net.get_openings()).item()

    loss_tracker.update_complete(position_loss, time_loss, latent_loss, loss_cur, loss_next, num_objects, openings, avg_gestalt, avg_gestalt2, avg_gestalt_mean, avg_update_gestalt, avg_update_position, num_bounded, lr, num_updates)
    pass

def log_modelsize(net):
    print(f'Loaded model with {sum([param.numel() for param in net.parameters()]):7d} parameters', flush=True)
    print(f'  States:     {sum([param.numel() for param in net.initial_states.parameters()]):7d} parameters', flush=True)
    print(f'  Encoder:    {sum([param.numel() for param in net.encoder.parameters()]):7d} parameters', flush=True)
    print(f'  Decoder:    {sum([param.numel() for param in net.decoder.parameters()]):7d} parameters', flush=True)
    print(f'  predictor:  {sum([param.numel() for param in net.predictor.parameters()]):7d} parameters', flush=True)
    print(f'  background: {sum([param.numel() for param in net.background.parameters()]):7d} parameters', flush=True)
    print("\n")
    pass

def initialize_loss_function(cfg):
    if not ('loss' in cfg.defaults):
        cfg.defaults.loss = 'bce'
        
    if cfg.defaults.loss == 'mse':
        imageloss = nn.MSELoss()
    elif cfg.defaults.loss == 'bce':
        imageloss = nn.BCELoss()
    else:
        raise NotImplementedError
    
    return imageloss
    
def init_optimizer(cfg, net):
    # backward compability:
    if not isinstance(cfg.learning_rate, dict):
        cfg.learning_rate = Dict({'lr': cfg.learning_rate})

    lr  = cfg.learning_rate.lr     
    optimizer_init = RAdam(net.initial_states.parameters(), lr = lr * 30)
    optimizer_encoder = RAdam(net.encoder.parameters(), lr = lr)
    optimizer_decoder = RAdam(net.decoder.parameters(), lr = lr)
    optimizer_predictor = RAdam(net.predictor.parameters(), lr = lr)
    optimizer_background = RAdam([net.background.mask], lr = lr)
    optimizer_update = RAdam(net.percept_gate_controller.parameters(), lr = lr)
    return optimizer_init,optimizer_encoder,optimizer_decoder,optimizer_predictor,optimizer_background,optimizer_update

def init_lr_scheduler_StepLR(cfg, *optimizer_list):
    scheduler_list = None
    if 'deacrease_lr_every' in cfg.learning_rate:
        print('Init lr scheduler')
        scheduler_list = []
        for optimizer in optimizer_list:
            scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.learning_rate.deacrease_lr_every, gamma=cfg.learning_rate.deacrease_lr_factor)
            scheduler_list.append(scheduler)
    return scheduler_list

def step_lr_scheduler(scheduler_list):
    if scheduler_list is not None:
        for scheduler in scheduler_list:
            scheduler.step()
    pass

def save_model(
    file,
    net, 
    optimizer_init, 
    optimizer_encoder, 
    optimizer_decoder, 
    optimizer_predictor,
    optimizer_background
):

    state = { }

    state['optimizer_init'] = optimizer_init.state_dict()
    state['optimizer_encoder'] = optimizer_encoder.state_dict()
    state['optimizer_decoder'] = optimizer_decoder.state_dict()
    state['optimizer_predictor'] = optimizer_predictor.state_dict()
    state['optimizer_background'] = optimizer_background.state_dict()

    state["model"] = net.state_dict()
    th.save(state, file)
    pass

def load_model(
    file,
    cfg,
    net, 
    optimizer_init, 
    optimizer_encoder, 
    optimizer_decoder, 
    optimizer_predictor,
    optimizer_background,
    load_optimizers = True,
    only_encoder_decoder = False
):
    device = th.device(cfg.device)
    state = th.load(file, map_location=device)
    print(f"load {file} to device {device}, only encoder/decoder: {only_encoder_decoder}")
    print(f"load optimizers: {load_optimizers}")

    if load_optimizers:
        optimizer_init.load_state_dict(state[f'optimizer_init'])
        for n in range(len(optimizer_init.param_groups)):
            optimizer_init.param_groups[n]['lr'] = cfg.learning_rate.lr

        optimizer_encoder.load_state_dict(state[f'optimizer_encoder'])
        for n in range(len(optimizer_encoder.param_groups)):
            optimizer_encoder.param_groups[n]['lr'] = cfg.learning_rate.lr

        optimizer_decoder.load_state_dict(state[f'optimizer_decoder'])
        for n in range(len(optimizer_decoder.param_groups)):
            optimizer_decoder.param_groups[n]['lr'] = cfg.learning_rate.lr

        optimizer_predictor.load_state_dict(state['optimizer_predictor'])
        for n in range(len(optimizer_predictor.param_groups)):
            optimizer_predictor.param_groups[n]['lr'] = cfg.learning_rate.lr

        optimizer_background.load_state_dict(state['optimizer_background'])
        for n in range(len(optimizer_background.param_groups)):
            optimizer_background.param_groups[n]['lr'] = cfg.model.background.learning_rate.lr

    # 1. Fill model with values of net 
    model = {}
    allowed_keys = []
    rand_state = net.state_dict()
    for key, value in rand_state.items():
        allowed_keys.append(key)
        model[key.replace(".module.", ".")] = value

    # 2. Overwrite with values from file
    for key, value in state["model"].items():
        # replace update_module with percept_gate_controller in key string:
        key = key.replace("update_module", "percept_gate_controller")

        if key in allowed_keys:
            if only_encoder_decoder:
                if ('encoder' in key) or ('decoder' in key):
                    model[key.replace(".module.", ".")] = value
            else:
                model[key.replace(".module.", ".")] = value

    net.load_state_dict(model)
        
    pass

def update_net_status(num_updates, net, cfg, optimizer_init):
    if num_updates == cfg.phases.background_pretraining_end and net.get_init_status() < 1:
        net.inc_init_level()

    if num_updates == cfg.phases.entity_pretraining_phase1_end and net.get_init_status() < 2:
        net.inc_init_level()

    if num_updates == cfg.phases.entity_pretraining_phase2_end and net.get_init_status() < 3:
        net.inc_init_level()
        for param in optimizer_init.param_groups:
            param['lr'] = cfg.learning_rate.lr

    pass
    
def plot_online(cfg, path, num_updates, input, background, mask, sequence_len, t, output_next, bg_error_next):
                
    # highlight error
    grayscale        = input[:,0:1] * 0.299 + input[:,1:2] * 0.587 + input[:,2:3] * 0.114
    object_mask_cur  = th.sum(mask[:,:-1], dim=1).unsqueeze(dim=1)
    highlited_input  = grayscale * (1 - object_mask_cur)
    highlited_input += grayscale * object_mask_cur * 0.3333333
    cmask = color_mask(mask[:,:-1])
    highlited_input  = highlited_input + cmask * 0.6666666

    input_ = rearrange(input[0], 'c h w -> h w c').detach().cpu()
    background_ = rearrange(background[0], 'c h w -> h w c').detach().cpu()
    mask_ = rearrange(mask[0,-1:], 'c h w -> h w c').detach().cpu()
    output_next_ = rearrange(output_next[0], 'c h w -> h w c').detach().cpu()
    bg_error_next_ = rearrange(bg_error_next[0], 'c h w -> h w c').detach().cpu()
    highlited_input_ = rearrange(highlited_input[0], 'c h w -> h w c').detach().cpu()

    if False:
        plot_path = os.path.join(path, 'plots', f'net_{num_updates}')
        os.makedirs(plot_path, exist_ok=True)
        cv2.imwrite(os.path.join(plot_path, f'input-{t+cfg.defaults.teacher_forcing:03d}.jpg'), input_.numpy() * 255)
        cv2.imwrite(os.path.join(plot_path, f'background-{t+cfg.defaults.teacher_forcing:03d}.jpg'), background_.numpy() * 255)
        cv2.imwrite(os.path.join(plot_path, f'error_mask-{t+cfg.defaults.teacher_forcing:03d}.jpg'), bg_error_next_.numpy() * 255)
        cv2.imwrite(os.path.join(plot_path, f'background_mask-{t+cfg.defaults.teacher_forcing:03d}.jpg'), mask_.numpy() * 255)
        cv2.imwrite(os.path.join(plot_path, f'output_next-{t+cfg.defaults.teacher_forcing:03d}.jpg'), output_next_.numpy() * 255)
        cv2.imwrite(os.path.join(plot_path, f'output_highlight-{t+cfg.defaults.teacher_forcing:03d}.jpg'), highlited_input_.numpy() * 255)

    # stack input, output, mask and highlight into one image horizontally
    img_tensor = th.cat([input_, highlited_input_, output_next_], dim=1)
    return img_tensor

def get_loader(cfg, set, cfg_net, shuffle = True, verbose=False):
    if ((cfg.datatype == 'bouncingballs') and verbose) or ((cfg.datatype == 'adept') and not shuffle):
        loader = DataLoader(
            set, 
            pin_memory = True, 
            num_workers = 0, 
            batch_size = cfg_net.batch_size, 
            shuffle = shuffle,
            drop_last = True, 
            persistent_workers = False
        )
    else:
        loader = DataLoader(
            set, 
            pin_memory = True, 
            num_workers = cfg.defaults.num_workers, 
            batch_size = cfg_net.batch_size, 
            shuffle = shuffle,
            drop_last = True, 
            prefetch_factor = cfg.defaults.prefetch_factor, 
            persistent_workers = True
        )
    return loader