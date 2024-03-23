import pickle
import torch as th
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn
import os
from scripts.utils.plot_utils import plot_timestep
from scripts.utils.eval_metrics import masks_to_boxes, pred_eval_step, postproc_mask
from scripts.utils.eval_utils import append_statistics, load_model, setup_result_folders, store_statistics
from scripts.utils.configuration import Configuration
from scripts.utils.io import init_device
import numpy as np
from einops import rearrange, repeat, reduce
from copy import deepcopy
import lpips
import torchvision.transforms as transforms

def evaluate(cfg: Configuration, dataset: Dataset, file, n, plot_frequency= 1, plot_first_samples = 0):

    # Set up cpu or gpu training
    device, verbose = init_device(cfg)

    # Config 
    cfg_net = cfg.model
    cfg_net.batch_size = 2 if verbose else 32
    cfg_net.batch_size = 1 if plot_first_samples > 0 else cfg_net.batch_size # only plotting first samples

    # Load model 
    net = load_model(cfg, cfg_net, file, device)
    net.eval()

    # config
    object_view = True
    individual_views = False
    root_path = None
    plotting_mode = (cfg_net.batch_size == 1) and (plot_first_samples > 0)

    # get evaluation sets
    set_test_array, evaluation_modes = get_evaluation_sets(dataset)

    # memory
    statistics_template = {'set': [], 'evalmode': [], 'scene': [], 'frame': [], 'image_error_mse': []}
    statistics_complete_slots = {'set': [], 'evalmode': [], 'scene': [], 'frame': [], 'slot':[], 'bound': [], 'slot_error': [], 'rawmask_size': [], 'alpha_pos': [], 'alpha_ges': []}
    metric_complete = None

    # Evaluation Specifics
    burn_in_length = 6
    rollout_length = 42
    cfg.defaults.skip_frames = 2
    blackout_p = 0.2
    target_size = (64,64)
    dataset.burn_in_length = burn_in_length
    dataset.rollout_length = rollout_length
    dataset.skip_length = cfg.defaults.skip_frames

    # Transformation utils
    to_small        = transforms.Resize(target_size)
    to_normalize    = transforms.Normalize((0.5, ), (0.5, ))
    to_smallnorm    = transforms.Compose([to_small, to_normalize])

    # Losses
    lpipsloss = lpips.LPIPS(net='vgg').to(device)
    mseloss = nn.MSELoss()
    
    for set_test in set_test_array:

        for evaluation_mode in evaluation_modes:
            print(f'Start evaluation loop: {evaluation_mode}')

            # load data
            dataloader = DataLoader(
                Subset(dataset, range(plot_first_samples)) if plotting_mode else dataset, 
                num_workers = 1, 
                pin_memory = False, 
                batch_size = cfg_net.batch_size,
                shuffle = False,
                drop_last = True,
            )

            # memory
            root_path, plot_path = setup_result_folders(file, n, set_test, evaluation_mode, object_view, individual_views)
            metric_complete = {'mse': [], 'ssim': [], 'psnr': [], 'percept_dist': [], 'ari': [], 'fari': [], 'miou': [], 'ap': [], 'ar': [], 'blackout': []}

            with th.no_grad():
                for i, input in enumerate(dataloader):
                    print(f'Processing sample {i+1}/{len(dataloader)}', flush=True)

                    # Load data
                    tensor = input[0].float().to(device)
                    background_fix = input[1].to(device)
                    gt_mask        = input[2].to(device)
                    gt_bbox        = input[3].to(device)
                    gt_pres_mask   = input[4].to(device)
                    sequence_len = tensor.shape[1]

                    # Placehodlers
                    mask_cur       = None
                    mask_last      = None
                    rawmask_last   = None
                    position_last  = None
                    gestalt_last   = None
                    priority_last  = None
                    slots_occlusionfactor = None
                    error_last  = None

                    # Memory
                    pred        = th.zeros((cfg_net.batch_size, rollout_length, 3, target_size[0], target_size[1])).to(device)
                    gt          = th.zeros((cfg_net.batch_size, rollout_length, 3, target_size[0], target_size[1])).to(device)
                    pred_mask   = th.zeros((cfg_net.batch_size, rollout_length, target_size[0], target_size[1])).to(device)
                    statistics_batch = deepcopy(statistics_template)

                    # Counters
                    num_rollout = 0
                    num_burnin  = 0
                    blackout_mem = [0]
 
                    # Loop through frames
                    for t_index,t in enumerate(range(-cfg.defaults.teacher_forcing, sequence_len-1)):

                        # Move to next frame
                        t_run = max(t, 0)
                        input  = tensor[:,t_run]
                        target = th.clip(tensor[:,t_run+1], 0, 1)

                        rollout_index = t_run - burn_in_length
                        if (rollout_index >= 0) and (evaluation_mode == 'blackout'):
                            blackout    = (th.rand(1) < blackout_p).float().to(device)
                            input       = blackout * (input * 0)         + (1-blackout) * input
                            error_last  = blackout * (error_last * 0)    + (1-blackout) * error_last
                            blackout_mem.append(blackout.int().cpu().item())

                        elif t>=0:
                            num_burnin += 1  

                        if (rollout_index >= 0) and (evaluation_mode != 'blackout'):  
                            blackout_mem.append(0)    

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
                            show_hidden = plotting_mode,
                            clean_slots = (t <= 0),
                        )

                        # 1. Track error for plots
                        if t >= 0:

                            # save prediction
                            if rollout_index >= -1:
                                pred[:,rollout_index+1] = to_smallnorm(output_next)
                                gt[:,rollout_index+1]   = to_smallnorm(target)
                                pred_mask[:,rollout_index+1] = postproc_mask(to_small(mask_next)[:,None,:,None])[:, 0]
                                num_rollout += 1

                            if plotting_mode and object_view:           
                                statistics_complete_slots, statistics_batch = compute_plot_statistics(cfg_net, statistics_complete_slots, mseloss, set_test, evaluation_mode, i, statistics_batch, t, target, output_next, mask_next, slots_bounded, slots_closed, rawmask_hidden)
            
                        # 2. Remember output
                        mask_last     = mask_next.clone()
                        rawmask_last  = rawmask_next.clone()
                        position_last = position_next.clone()
                        gestalt_last  = gestalt_next.clone()
                        priority_last = priority_next.clone()
                        
                        # 3. Error for next frame
                        bg_error_next = th.sqrt(reduce((target - background)**2, 'b c h w -> b 1 h w', 'mean')).detach()

                        # prediction error
                        error_next    = th.sqrt(reduce((target - output_next)**2, 'b c h w -> b 1 h w', 'mean')).detach()
                        error_next    = th.sqrt(error_next) * bg_error_next
                        error_last    = error_next.clone()

                        # 4. plot preparation
                        if plotting_mode and (t % plot_frequency == 0):
                            plot_timestep(cfg, cfg_net, input, target, mask_cur, mask_next, output_next, position_encoder_cur, position_next, rawmask_hidden, rawmask_cur, rawmask_next, largest_object, object_cur, object_next, object_hidden, slots_bounded, slots_partially_occluded_cur, slots_occluded_cur, slots_partially_occluded_next, slots_occluded_next, slots_closed, None, None, error_next, output_hidden, object_view, individual_views, statistics_complete_slots, statistics_batch, sequence_len, root_path, plot_path, t_index, t, i)
        
                    # Compute prediction accuracy based on Slotformer metrics (ARI, FARI, mIoU, AP, AR)
                    if not plotting_mode:
                        for b in range(cfg_net.batch_size):
                            metric_dict = pred_eval_step(
                                gt=gt[b:b+1],
                                pred=pred[b:b+1],
                                pred_mask=pred_mask.long()[b:b+1],
                                pred_bbox=masks_to_boxes(pred_mask.long()[b:b+1], cfg_net.num_objects+1),
                                gt_mask=gt_mask.long()[b:b+1],
                                gt_pres_mask=gt_pres_mask[b:b+1], 
                                gt_bbox=gt_bbox[b:b+1],
                                lpips_fn=lpipsloss,
                                eval_traj=True,
                            )
                            metric_dict['blackout'] = blackout_mem
                            metric_complete = append_statistics(metric_dict, metric_complete)

                    # sanity check
                    if (num_rollout != rollout_length) and (num_burnin != burn_in_length) and (evaluation_mode == 'rollout'):
                        raise ValueError('Number of rollout steps and burnin steps must be equal to the sequence length.')

            if not plotting_mode:
                average_dic = compute_statistics_summary(metric_complete, evaluation_mode, root_path=root_path)

                # Store statistics
                with open(os.path.join(f'{root_path}/statistics', f'{evaluation_mode}_metric_complete.pkl'), 'wb') as f:
                    pickle.dump(metric_complete, f)
                with open(os.path.join(f'{root_path}/statistics', f'{evaluation_mode}_metric_average.pkl'), 'wb') as f:
                    pickle.dump(average_dic, f)

    print('-- Evaluation Done --')
    if object_view and os.path.exists(f'{root_path}/tmp.jpg'):
        os.remove(f'{root_path}/tmp.jpg')
    pass

def compute_statistics_summary(metric_complete, evaluation_mode, root_path=None, consider_first_n_frames = None):
    string = ''
    def add_text(string, text, last=False):
        string = string + ' \n ' + text
        return string

    average_dic = {}
    if consider_first_n_frames is not None:
        for key in metric_complete:
            for sample in range(len(metric_complete[key])):
                metric_complete[key][sample] = metric_complete[key][sample][:consider_first_n_frames]

    for key in metric_complete:
                    # take average over all frames
        average_dic[key + '_complete_average']   = np.mean(metric_complete[key])
        average_dic[key + '_complete_std']       = np.std(metric_complete[key])
        average_dic[key + '_complete_sum']       = np.sum(np.mean(metric_complete[key], axis=0)) # checked with GSWM code!
        string = add_text(string, f'{key} complete average: {average_dic[key + "_complete_average"]:.4f} +/- {average_dic[key + "_complete_std"]:.4f} (sum: {average_dic[key + "_complete_sum"]:.4f})')
        #print(f'{key} complete average: {average_dic[key + "complete_average"]:.4f} +/- {average_dic[key + "complete_std"]:.4f} (sum: {average_dic[key + "complete_sum"]:.4f})')

        if evaluation_mode == 'blackout':
            # take average only for frames where blackout occurs
            blackout_mask = np.array(metric_complete['blackout']) > 0
            average_dic[key + '_blackout_average'] = np.mean(np.array(metric_complete[key])[blackout_mask])
            average_dic[key + '_blackout_std']     = np.std(np.array(metric_complete[key])[blackout_mask])
            average_dic[key + '_visible_average']  = np.mean(np.array(metric_complete[key])[blackout_mask == False])
            average_dic[key + '_visible_std']      = np.std(np.array(metric_complete[key])[blackout_mask == False])
    
            #print(f'{key} blackout average: {average_dic[key + "blackout_average"]:.4f} +/- {average_dic[key + "blackout_std"]:.4f}')
            #print(f'{key} visible average: {average_dic[key + "visible_average"]:.4f} +/- {average_dic[key + "visible_std"]:.4f}')
            string = add_text(string, f'{key} blackout average: {average_dic[key + "_blackout_average"]:.4f} +/- {average_dic[key + "_blackout_std"]:.4f}')
            string = add_text(string, f'{key} visible average: {average_dic[key + "_visible_average"]:.4f} +/- {average_dic[key + "_visible_std"]:.4f}')

    print(string)
    if root_path is not None:
        f'{root_path}/statistics', f'{evaluation_mode}_metric_complete.pkl'
        with open(os.path.join(f'{root_path}/statistics', f'{evaluation_mode}_metric_average.txt'), 'w') as f:
            f.write(string)

    return average_dic

def compute_plot_statistics(cfg_net, statistics_complete_slots, mseloss, set_test, evaluation_mode, i, statistics_batch, t, target, output_next, mask_next, slots_bounded, slots_closed, rawmask_hidden):
    statistics_batch = store_statistics(statistics_batch,
                                                                set_test['type'],
                                                                evaluation_mode,
                                                                set_test['samples'][i],
                                                                t,
                                                                mseloss(output_next, target).item()
                                                                )

    # compute slot-wise prediction error
    output_slot = repeat(mask_next[:,:-1], 'b o h w -> b o 3 h w') * repeat(output_next, 'b c h w -> b o c h w', o=cfg_net.num_objects)
    target_slot = repeat(mask_next[:,:-1], 'b o h w -> b o 3 h w') * repeat(target, 'b c h w -> b o c h w', o=cfg_net.num_objects)
    slot_error = reduce((output_slot - target_slot)**2, 'b o c h w -> b o', 'mean')

    # compute rawmask_size
    rawmask_size = reduce(rawmask_hidden[:, :-1], 'b o h w-> b o', 'sum')

    statistics_complete_slots = store_statistics(statistics_complete_slots,
                                                                            [set_test['type']] * cfg_net.num_objects,
                                                                            [evaluation_mode] * cfg_net.num_objects,
                                                                            [set_test['samples'][i]] * cfg_net.num_objects,
                                                                            [t] * cfg_net.num_objects,
                                                                            range(cfg_net.num_objects),
                                                                            slots_bounded.cpu().numpy().flatten().astype(int),
                                                                            slot_error.cpu().numpy().flatten(),
                                                                            rawmask_size.cpu().numpy().flatten(),
                                                                            slots_closed[:, :, 1].cpu().numpy().flatten(),
                                                                            slots_closed[:, :, 0].cpu().numpy().flatten(),
                                                                            extend = True)
                                                
    return statistics_complete_slots,statistics_batch

def get_evaluation_sets(dataset):

    set = {"samples": np.arange(len(dataset), dtype=int), "type": "test"}
    evaluation_modes = ['blackout', 'open'] # use 'open' for no blackouts
    set_test_array = [set]

    return set_test_array, evaluation_modes
