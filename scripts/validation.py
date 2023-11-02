import torch as th
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from einops import rearrange, repeat, reduce
from scripts.utils.configuration import Configuration
from model.loci import Loci
import time
import lpips
from skimage.metrics import structural_similarity as ssimloss
from skimage.metrics import peak_signal_noise_ratio as psnrloss

def validation_adept(valloader: DataLoader, net: Loci, cfg: Configuration, device):

    # memory
    mseloss = nn.MSELoss()
    avgloss = 0
    start_time = time.time()

    with th.no_grad():
        for i, input in enumerate(valloader):

            # get input frame and target frame
            tensor = input[0].float().to(device)
            background_fix  = input[1].to(device)

            # apply skip frames
            tensor = tensor[:,range(0, tensor.shape[1], cfg.defaults.skip_frames)]
            sequence_len = tensor.shape[1]

            # initial frame
            input  = tensor[:,0]
            target = th.clip(tensor[:,0], 0, 1)
            error_last  = None

            # placehodlers
            mask_cur       = None
            mask_last      = None
            rawmask_last   = None
            position_last  = None
            gestalt_last   = None
            priority_last  = None
            gt_positions_target = None
            slots_occlusionfactor = None

            # loop through frames
            for t_index,t in enumerate(range(-cfg.defaults.teacher_forcing, sequence_len-1)):

                # move to next frame
                t_run = max(t, 0)
                input  = tensor[:,t_run]
                target = th.clip(tensor[:,t_run+1], 0, 1)

                # obtain prediction
                (
                    output_next, 
                    position_next, 
                    gestalt_next, 
                    priority_next, 
                    mask_next, 
                    rawmask_next,
                    object_next, 
                    background, 
                    slots_occlusionfactor,
                    output_cur,
                    position_cur,
                    gestalt_cur,
                    priority_cur,
                    mask_cur,
                    rawmask_cur,
                    object_cur,
                    position_encoder_cur,
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
                ) = net(
                    input, 
                    error_last,
                    mask_last, 
                    rawmask_last,
                    position_last, 
                    gestalt_last,
                    priority_last,
                    background_fix,
                    slots_occlusionfactor,
                    reset = (t == -cfg.defaults.teacher_forcing),
                    evaluate=True,
                    warmup = (t < 0),
                    shuffleslots = False,
                    reset_mask = (t <= 0),
                    allow_spawn = True,
                    show_hidden = False,
                    clean_slots = (t <= 0),
                )

                # 1. Track error
                if t >= 0:
                    loss = mseloss(output_next, target)
                    avgloss += loss.item()

                # 2. Remember output
                mask_last     = mask_next.clone()
                rawmask_last  = rawmask_next.clone()
                position_last = position_next.clone()
                gestalt_last  = gestalt_next.clone()
                priority_last = priority_next.clone()
                        
                # 3. Error for next frame
                # background error
                bg_error_cur  = th.sqrt(reduce((input - background)**2, 'b c h w -> b 1 h w', 'mean')).detach()
                bg_error_next = th.sqrt(reduce((target - background)**2, 'b c h w -> b 1 h w', 'mean')).detach()

                # prediction error
                error_next    = th.sqrt(reduce((target - output_next)**2, 'b c h w -> b 1 h w', 'mean')).detach()
                error_next    = th.sqrt(error_next) * bg_error_next
                error_last    = error_next.clone()

    print(f"Validation loss: {avgloss / len(valloader.dataset):.2e}, Time: {time.time() - start_time}")
            
    pass


def validation_clevrer(valloader: DataLoader, net: Loci, cfg: Configuration, device):

    # memory
    mseloss = nn.MSELoss()
    lpipsloss = lpips.LPIPS(net='vgg').to(device)
    avgloss_mse = 0
    avgloss_lpips = 0
    avgloss_psnr = 0
    avgloss_ssim = 0
    start_time = time.time()

    burn_in_length = 6
    rollout_length = 42

    with th.no_grad():
        for i, input in enumerate(valloader):

            # get input frame and target frame
            tensor = input[0].float().to(device)
            background_fix  = input[1].to(device)

            # apply skip frames
            tensor = tensor[:,range(0, tensor.shape[1], cfg.defaults.skip_frames)]
            sequence_len = tensor.shape[1]

            # initial frame
            input  = tensor[:,0]
            target = th.clip(tensor[:,0], 0, 1)
            error_last  = None

            # placehodlers
            mask_cur       = None
            mask_last      = None
            rawmask_last   = None
            position_last  = None
            gestalt_last   = None
            priority_last  = None
            gt_positions_target = None
            slots_occlusionfactor = None

            # loop through frames
            for t_index,t in enumerate(range(-cfg.defaults.teacher_forcing, min(burn_in_length + rollout_length-1, sequence_len-1))):

                # move to next frame
                t_run = max(t, 0)
                input  = tensor[:,t_run]
                target = th.clip(tensor[:,t_run+1], 0, 1)
                if t_run >= burn_in_length:
                    blackout    = th.tensor((np.random.rand(valloader.batch_size) < 0.2)[:,None,None,None]).float().to(device)
                    input       = blackout * (input * 0)         + (1-blackout) * input
                    error_last  = blackout * (error_last * 0)    + (1-blackout) * error_last

                # obtain prediction
                (
                    output_next, 
                    position_next, 
                    gestalt_next, 
                    priority_next, 
                    mask_next, 
                    rawmask_next,
                    object_next, 
                    background, 
                    slots_occlusionfactor,
                    output_cur,
                    position_cur,
                    gestalt_cur,
                    priority_cur,
                    mask_cur,
                    rawmask_cur,
                    object_cur,
                    position_encoder_cur,
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
                ) = net(
                    input, 
                    error_last,
                    mask_last, 
                    rawmask_last,
                    position_last, 
                    gestalt_last,
                    priority_last,
                    background_fix,
                    slots_occlusionfactor,
                    reset = (t == -cfg.defaults.teacher_forcing),
                    evaluate=True,
                    warmup = (t < 0),
                    shuffleslots = False,
                    reset_mask = (t <= 0),
                    allow_spawn = True,
                    show_hidden = False,
                    clean_slots = (t <= 0),
                )

                # 1. Track error
                if t >= 0:
                    loss_mse    = mseloss(output_next, target)
                    loss_ssim   = np.sum([ssimloss(output_next[i].cpu().numpy(), target[i].cpu().numpy(), channel_axis=0,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,data_range=1) for i in range(output_next.shape[0])]),
                    loss_psnr   = np.sum([psnrloss(output_next[i].cpu().numpy(), target[i].cpu().numpy(), data_range=1)  for i in range(output_next.shape[0])]),
                    loss_lpips  = th.sum(lpipsloss(output_next*2-1, target*2-1))

                    avgloss_mse += loss_mse.item()
                    avgloss_ssim += loss_ssim[0].item()
                    avgloss_psnr += loss_psnr[0].item()
                    avgloss_lpips += loss_lpips.item()

                # 2. Remember output
                mask_last     = mask_next.clone()
                rawmask_last  = rawmask_next.clone()
                position_last = position_next.clone()
                gestalt_last  = gestalt_next.clone()
                priority_last = priority_next.clone()
                        
                # 3. Error for next frame
                # background error
                bg_error_next = th.sqrt(reduce((target - background)**2, 'b c h w -> b 1 h w', 'mean')).detach()

                # prediction error
                error_next    = th.sqrt(reduce((target - output_next)**2, 'b c h w -> b 1 h w', 'mean')).detach()
                error_next    = th.sqrt(error_next) * bg_error_next
                error_last    = error_next.clone()

    print(f"MSE loss: {avgloss_mse / len(valloader.dataset):.2e}, LPIPS loss: {avgloss_lpips / len(valloader.dataset):.2e}, PSNR loss: {avgloss_psnr / len(valloader.dataset):.2e}, SSIM loss: {avgloss_ssim / len(valloader.dataset):.2e}, Time: {time.time() - start_time}")
            
    pass