import os
import torch as th
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from einops import rearrange, repeat, reduce
from scripts.evaluation_bb import distance_eval_step
from scripts.evaluation_clevrer import compute_statistics_summary
from scripts.utils.configuration import Configuration
from model.loci import Loci
import time
import lpips
from skimage.metrics import structural_similarity as ssimloss
from skimage.metrics import peak_signal_noise_ratio as psnrloss

from scripts.utils.eval_metrics import masks_to_boxes, postproc_mask, pred_eval_step
from scripts.utils.eval_utils import append_statistics, compute_position_from_mask
from scripts.utils.plot_utils import plot_timestep

def validation_adept(valloader: DataLoader, net: Loci, cfg: Configuration, device, writer, epoch, root_path):

    # memory
    mseloss = nn.MSELoss()
    loss_next = 0
    start_time = time.time()
    cfg_net = cfg.model
    num_steps = 0
    plot_path = os.path.join(root_path, 'plots', f'epoch_{epoch}')

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
                    loss_next += loss.item()
                    num_steps += 1

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

                # PLotting
                if (i == 0) and (t_index % 2 == 0) and (epoch % 3 == 0):
                    os.makedirs(os.path.join(plot_path,  'object'), exist_ok=True)
                    openings = net.get_openings()
                    img_tensor = plot_timestep(cfg, cfg_net, input, target, mask_cur, mask_next, output_next, position_encoder_cur, position_next, rawmask_hidden, rawmask_cur, rawmask_next, largest_object, object_cur, object_next, object_hidden, slots_bounded, slots_partially_occluded_cur, slots_occluded_cur, slots_partially_occluded_next, slots_occluded_next, slots_closed, None, None, error_next, None, True, False, None, None, sequence_len, root_path, plot_path, t_index, t, i, openings=openings)

    print(f"Validation loss: {loss_next / num_steps:.2e}, Time: {time.time() - start_time}")
    writer.add_scalar('Val/Prediction Loss', loss_next / num_steps, epoch)

    pass


def validation_clevrer(valloader: DataLoader, net: Loci, cfg: Configuration, device, writer, epoch, root_path):

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
    
    plot_path = os.path.join(root_path, 'plots', f'epoch_{epoch}')
    os.makedirs(os.path.join(plot_path,  'object'), exist_ok=True)

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

                # PLotting
                if i == 0:
                    openings = net.get_openings()
                    img_tensor = plot_timestep(cfg, cfg.model, input, target, mask_cur, mask_next, output_next, position_encoder_cur, position_next, rawmask_hidden, rawmask_cur, rawmask_next, largest_object, object_cur, object_next, object_hidden, slots_bounded, slots_partially_occluded_cur, slots_occluded_cur, slots_partially_occluded_next, slots_occluded_next, slots_closed, None, None, error_next, None, True, False, None, None, sequence_len, root_path, plot_path, t_index, t, i, openings=openings)


    print(f"MSE loss: {avgloss_mse / len(valloader.dataset):.2e}, LPIPS loss: {avgloss_lpips / len(valloader.dataset):.2e}, PSNR loss: {avgloss_psnr / len(valloader.dataset):.2e}, SSIM loss: {avgloss_ssim / len(valloader.dataset):.2e}, Time: {time.time() - start_time}")
    writer.add_scalar('Val/Prediction Loss', avgloss_mse / len(valloader.dataset), epoch)
    writer.add_scalar('Val/LPIPS Loss', avgloss_lpips / len(valloader.dataset), epoch)
    writer.add_scalar('Val/PSNR Loss', avgloss_psnr / len(valloader.dataset), epoch)
    writer.add_scalar('Val/SSIM Loss', avgloss_ssim / len(valloader.dataset), epoch)

    pass


def validation_bb(valloader: DataLoader, net: Loci, cfg: Configuration, device, writer, epoch, root_path):

    # memory
    start_time = time.time()
    net.eval()
    evaluation_mode = 'vidpred_black'
    use_meds = True

    # Evaluation Specifics
    burn_in_length = 10
    rollout_length = 20
    rollout_length_stats = 10 # only consider the first 10 frames for statistics
    target_size = (64, 64)

    # Losses
    lpipsloss = lpips.LPIPS(net='vgg').to(device)
    mseloss = nn.MSELoss()
    metric_complete = {'mse': [], 'ssim': [], 'psnr': [], 'percept_dist': [], 'ari': [], 'fari': [], 'miou': [], 'ap': [], 'ar': [], 'meds': [], 'ari_hidden': [], 'fari_hidden': [], 'miou_hidden': []}
    loss_next = 0.0
    loss_cur  = 0.0
    num_steps = 0
    plot_path = os.path.join(root_path, 'plots', f'epoch_{epoch}')
    os.makedirs(os.path.join(plot_path,  'object'), exist_ok=True)

    with th.no_grad():
        for i, input in enumerate(valloader):

            # Load data
            tensor = input[0].float().to(device)
            background_fix  = input[1].to(device)
            gt_pos          = input[2].to(device)
            gt_mask         = input[3].to(device)
            gt_pres_mask    = input[4].to(device)
            gt_hidden_mask  = input[5].to(device)
            sequence_len    = tensor.shape[1]

            # placehodlers
            mask_cur       = None
            mask_last      = None
            rawmask_last   = None
            position_last  = None
            gestalt_last   = None
            priority_last  = None
            gt_positions_target = None
            slots_occlusionfactor = None
            error_last  = None

            # Memory
            cfg_net = cfg.model
            num_objects_bb = gt_pos.shape[2]
            pred_pos_batch        = th.zeros((cfg_net.batch_size, rollout_length, num_objects_bb, 2)).to(device)
            gt_pos_batch          = th.zeros((cfg_net.batch_size, rollout_length, num_objects_bb, 2)).to(device)
            pred_img_batch        = th.zeros((cfg_net.batch_size, rollout_length, 3, target_size[0], target_size[1])).to(device)
            gt_img_batch          = th.zeros((cfg_net.batch_size, rollout_length, 3, target_size[0], target_size[1])).to(device)
            pred_mask_batch       = th.zeros((cfg_net.batch_size, rollout_length, target_size[0], target_size[1])).to(device)
            pred_hidden_mask_batch   = th.zeros((cfg_net.batch_size, rollout_length, target_size[0], target_size[1])).to(device)
            
            # Counters
            num_rollout = 0
            num_burnin  = 0
            
            # loop through frames
            for t_index,t in enumerate(range(-cfg.defaults.teacher_forcing, burn_in_length+rollout_length)):

                # Move to next frame
                t_run = max(t, 0)
                input  = tensor[:,t_run]
                target_cur  = tensor[:,t_run]
                target = th.clip(tensor[:,t_run+1], 0, 1)
                gt_pos_t = gt_pos[:,t_run+1]/32-1
                gt_pos_t = th.concat((gt_pos_t, th.ones_like(gt_pos_t[:,:,:1])), dim=2)

                rollout_index = t_run - burn_in_length
                rollout_active = False
                if t>=0:
                    if rollout_index >= 0:
                        num_rollout += 1
                        if (evaluation_mode == 'vidpred_black'):
                            input = output_next * 0
                            error_last = error_last * 0
                            rollout_active = True
                        elif (evaluation_mode == 'vidpred_auto'):
                            input = output_next
                            error_last = error_last * 0
                            rollout_active = True
                    else:
                        num_burnin += 1  

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
                    shuffleslots = True,
                    reset_mask = (t <= 0),
                    allow_spawn = True,
                    show_hidden = False,
                    clean_slots = False,
                )

                # 1. Track error
                if t >= 0:

                    if (rollout_index >= 0):
                        # store positions per batch
                        if use_meds:
                            if False:
                                pred_pos_batch[:,rollout_index] = rearrange(position_next, 'b (o c) -> b o c', o=cfg_net.num_objects)[:,:,:2]
                            else:
                                pred_pos_batch[:,rollout_index] = compute_position_from_mask(rawmask_next)

                            gt_pos_batch[:,rollout_index] = gt_pos_t[:,:,:2]

                        pred_img_batch[:,rollout_index] = output_next
                        gt_img_batch[:,rollout_index] = target

                        # Here we compute only the foreground segmentation mask
                        pred_mask_batch[:,rollout_index] = postproc_mask(mask_next[:,None,:,None])[:, 0]

                        # Here we compute the hidden segmentation
                        occluded_cur        = th.clip(rawmask_next - mask_next, 0, 1)[:,:-1]
                        occluded_sum_cur    = 1-(reduce(occluded_cur, 'b c h w -> b h w', 'max') > 0.5).float()
                        occluded_cur        = th.cat((occluded_cur, occluded_sum_cur[:,None]), dim=1)
                        pred_hidden_mask_batch[:,rollout_index] = postproc_mask(occluded_cur[:,None,:,None])[:, 0]

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

                # Prediction and encoder Loss
                loss_next += mseloss(output_next * bg_error_next, target * bg_error_next)
                loss_cur  += mseloss(output_cur * bg_error_cur,   input  * bg_error_cur)
                num_steps += 1

                # PLotting
                if i == 0:
                    openings = net.get_openings()
                    img_tensor = plot_timestep(cfg, cfg_net, input, target_cur, mask_cur, mask_next, output_next, position_encoder_cur, position_next, rawmask_hidden, rawmask_cur, rawmask_next, largest_object, object_cur, object_next, object_hidden, slots_bounded, slots_partially_occluded_cur, slots_occluded_cur, slots_partially_occluded_next, slots_occluded_next, slots_closed, gt_pos_t, None, error_next, None, True, False, None, None, sequence_len, root_path, plot_path, t_index, t, i, rollout_mode=rollout_active, openings=openings)

        for b in range(cfg_net.batch_size):

            # perceptual similarity from slotformer paper
            metric_dict = pred_eval_step(
                gt              = gt_img_batch[b:b+1],
                pred            = pred_img_batch[b:b+1],
                pred_mask       = pred_mask_batch.long()[b:b+1],
                pred_mask_hidden = pred_hidden_mask_batch.long()[b:b+1],
                pred_bbox       = None,
                gt_mask         = gt_mask.long()[b:b+1, burn_in_length+1:burn_in_length+rollout_length+1],
                gt_mask_hidden  = gt_hidden_mask.long()[b:b+1, burn_in_length+1:burn_in_length+rollout_length+1],
                gt_pres_mask    = gt_pres_mask[b:b+1, burn_in_length+1:burn_in_length+rollout_length+1], 
                gt_bbox         = None,
                lpips_fn        = lpipsloss,
                eval_traj       = True,
            )

            metric_dict['meds'] = distance_eval_step(gt_pos_batch[b], pred_pos_batch[b])
            metric_complete = append_statistics(metric_dict, metric_complete)

        # sanity check
        if (num_rollout != rollout_length) and (num_burnin != burn_in_length):
            raise ValueError('Number of rollout steps and burnin steps must be equal to the sequence length.')
        
    dic = compute_statistics_summary(metric_complete, evaluation_mode, consider_first_n_frames=rollout_length_stats)
    writer.add_scalar('Val/Meds', dic['meds_complete_sum'], epoch)

    writer.add_scalar('Val/ARI_hidden', dic['ari_hidden_complete_average'], epoch)
    writer.add_scalar('Val/ARI', dic['ari_complete_average'], epoch)
    writer.add_scalar('Val/LPIPS', dic['percept_dist_complete_average'], epoch)

    writer.add_scalar('Val/Prediction Loss', loss_next / num_steps, epoch)
    writer.add_scalar('Val/Encoding Loss', loss_cur / num_steps, epoch)

    net.train()
    pass