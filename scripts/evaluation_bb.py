import pickle
import cv2
import torch as th
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn
import os
from scripts.evaluation_adept import calculate_tracking_error
from scripts.evaluation_clevrer import compute_statistics_summary
from scripts.utils.plot_utils import plot_timestep
from scripts.utils.eval_metrics import masks_to_boxes, pred_eval_step, postproc_mask
from scripts.utils.eval_utils import append_statistics, compute_position_from_mask, load_model, setup_result_folders, store_statistics
from scripts.utils.configuration import Configuration
from scripts.utils.io import init_device
import numpy as np
from einops import rearrange, repeat, reduce
from copy import deepcopy
import lpips
import torchvision.transforms as transforms
import motmetrics as mm

def evaluate(cfg: Configuration, dataset: Dataset, file, n, plot_first_samples = 0):

    # Set up cpu or gpu training
    device, verbose = init_device(cfg)

    # Config 
    cfg_net = cfg.model
    cfg_net.batch_size = 2 if verbose else 32
    #cfg_net.num_objects = 3
    cfg_net.inner_loop_enabled = True
    if 'num_objects_test' in cfg_net:
        cfg_net.num_objects = cfg_net.num_objects_test
    dataset = Subset(dataset, range(4)) if verbose else dataset
    
    # Load model 
    net = load_model(cfg, cfg_net, file, device)
    net.eval()
    net.predictor.enable_att_weights()

    # config
    object_view = True
    individual_views = False
    root_path = None
    use_meds = True

    # get evaluation sets
    set_test_array, evaluation_modes = get_evaluation_sets(dataset)

    # memory
    statistics_template = {'set': [], 'evalmode': [], 'scene': [], 'frame': [], 'image_error_mse': []}
    statistics_complete_slots = {'set': [], 'evalmode': [], 'scene': [], 'frame': [], 'slot':[], 'bound': [], 'slot_error': [], 'rawmask_size': [], 'alpha_pos': [], 'alpha_ges': []}
    metric_complete = None 

    # Evaluation Specifics
    burn_in_length = 10
    rollout_length = 90
    rollout_length_stats = 10 # only consider the first 10 frames for statistics
    target_size = (64, 64)

    # Losses
    lpipsloss = lpips.LPIPS(net='vgg').to(device)
    mseloss = nn.MSELoss()
    
    for set_test in set_test_array:
        
        for evaluation_mode in evaluation_modes:
            print(f'Start evaluation loop: {evaluation_mode}')

            # load data
            dataloader = DataLoader(
                dataset, 
                num_workers = 0, 
                pin_memory = False, 
                batch_size = cfg_net.batch_size,
                shuffle = False,
                drop_last = True,
            )

            # memory
            root_path, plot_path = setup_result_folders(file, n, set_test, evaluation_mode, object_view, individual_views)
            metric_complete = {'mse': [], 'ssim': [], 'psnr': [], 'percept_dist': [], 'ari': [], 'fari': [], 'miou': [], 'ap': [], 'ar': [], 'meds': [], 'ari_hidden': [], 'fari_hidden': [], 'miou_hidden': []}
            video_list = []

            # set seed: if there is a number in the evaluation mode, use it as seed
            plot_mode  = True
            if evaluation_mode[-1].isdigit():
                seed = int(evaluation_mode[-1])
                th.manual_seed(seed)
                np.random.seed(seed)
                print(f'Set seed to {seed}')
                if int(evaluation_mode[-1]) > 1:
                    plot_mode = False

            with th.no_grad():
                for i, input in enumerate(dataloader):
                    print(f'Processing sample {i+1}/{len(dataloader)}', flush=True)

                    # Load data
                    tensor = input[0].float().to(device)
                    background_fix  = input[1].to(device)
                    gt_pos          = input[2].to(device)
                    gt_mask         = input[3].to(device)
                    gt_pres_mask    = input[4].to(device)
                    gt_hidden_mask  = input[5].to(device)
                    sequence_len    = tensor.shape[1]

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
                    statistics_batch = deepcopy(statistics_template)
                    pred_pos_batch        = th.zeros((cfg_net.batch_size, rollout_length, cfg_net.num_objects, 2)).to(device)
                    gt_pos_batch          = th.zeros((cfg_net.batch_size, rollout_length, cfg_net.num_objects, 2)).to(device)
                    pred_img_batch        = th.zeros((cfg_net.batch_size, rollout_length, 3, target_size[0], target_size[1])).to(device)
                    gt_img_batch          = th.zeros((cfg_net.batch_size, rollout_length, 3, target_size[0], target_size[1])).to(device)
                    pred_mask_batch       = th.zeros((cfg_net.batch_size, rollout_length, target_size[0], target_size[1])).to(device)
                    pred_hidden_mask_batch       = th.zeros((cfg_net.batch_size, rollout_length, target_size[0], target_size[1])).to(device)

                    # Counters
                    num_rollout = 0
                    num_burnin  = 0
 
                    # Loop through frames
                    for t_index,t in enumerate(range(-cfg.defaults.teacher_forcing, sequence_len-1)):

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
                                if ('vidpred_black' in evaluation_mode):
                                    input = output_next * 0
                                    rollout_active = True
                                elif ('vidpred_auto' in evaluation_mode):
                                    input = output_next
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

                        # 1. Track error for plots
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
                        bg_error_next = th.sqrt(reduce((target - background)**2, 'b c h w -> b 1 h w', 'mean')).detach()

                        # prediction error
                        error_next    = th.sqrt(reduce((target - output_next)**2, 'b c h w -> b 1 h w', 'mean')).detach()
                        error_next    = th.sqrt(error_next) * bg_error_next
                        error_last    = error_next.clone()

                        # PLotting
                        if i == 0 and plot_mode:
                            att = net.predictor.get_att_weights()
                            openings = net.get_openings()
                            img_tensor = plot_timestep(cfg, cfg_net, input, target_cur, mask_cur, mask_next, output_next, position_encoder_cur, position_next, rawmask_hidden, rawmask_cur, rawmask_next, largest_object, object_cur, object_next, object_hidden, slots_bounded, slots_partially_occluded_cur, slots_occluded_cur, slots_partially_occluded_next, slots_occluded_next, slots_closed, None, None, error_next, None, True, individual_views, None, None, sequence_len, root_path, None, t_index, t, i, rollout_mode=rollout_active, num_vid=plot_first_samples, att= att, openings=None)
                            video_list.append(img_tensor)
                            
                    # log video
                    if i == 0 and plot_mode:
                        video_tensor = rearrange(th.stack(video_list, dim=0), 't b c h w -> b t h w c')
                        save_videos(video_tensor, f'{plot_path}/object', verbose=verbose, trace_plot=True)

                    # Compute prediction accuracy based on Slotformer metrics (ARI, FARI, mIoU, AP, AR)
                    for b in range(cfg_net.batch_size):

                        # perceptual similarity from slotformer paper
                        metric_dict = pred_eval_step(
                            gt              = gt_img_batch[b:b+1],
                            pred            = pred_img_batch[b:b+1],
                            pred_mask       = pred_mask_batch.long()[b:b+1],
                            pred_mask_hidden = pred_hidden_mask_batch.long()[b:b+1],
                            pred_bbox       = None,
                            gt_mask         = gt_mask.long()[b:b+1, burn_in_length+1:],
                            gt_mask_hidden  = gt_hidden_mask.long()[b:b+1, burn_in_length+1:],
                            gt_pres_mask    = gt_pres_mask[b:b+1, burn_in_length+1:], 
                            gt_bbox         = None,
                            lpips_fn        = lpipsloss,
                            eval_traj       = True,
                        )

                        metric_dict['meds'] = distance_eval_step(gt_pos_batch[b], pred_pos_batch[b])
                        metric_complete = append_statistics(metric_dict, metric_complete)

                    # sanity check
                    if (num_rollout != rollout_length) and (num_burnin != burn_in_length) and ('vidpred' in evaluation_mode):
                        raise ValueError('Number of rollout steps and burnin steps must be equal to the sequence length.')
                    

            average_dic = compute_statistics_summary(metric_complete, evaluation_mode, root_path=root_path, consider_first_n_frames=rollout_length_stats)

            # Store statistics
            with open(os.path.join(f'{root_path}/statistics', f'{evaluation_mode}_metric_complete.pkl'), 'wb') as f:
                pickle.dump(metric_complete, f)
            with open(os.path.join(f'{root_path}/statistics', f'{evaluation_mode}_metric_average.pkl'), 'wb') as f:
                pickle.dump(average_dic, f)

    print('-- Evaluation Done --')
    if object_view and os.path.exists(f'{root_path}/tmp.jpg'):
        os.remove(f'{root_path}/tmp.jpg')
    pass

# store videos as jpgs and then use ffmpeg to convert to video
def save_videos(video_tensor, plot_path, verbose=False, fps=10, trace_plot=False):
    video_tensor = video_tensor.cpu().numpy()
    img_path = plot_path + '/img'
    for b in range(video_tensor.shape[0]):
        os.makedirs(img_path, exist_ok=True)
        video = video_tensor[b]
        video = (video).astype(np.uint8)
        for t in range(video.shape[0]):
            cv2.imwrite(f'{img_path}/{b}_{t:04d}.jpg', video[t])

        if verbose:
            os.system(f"ffmpeg -r {fps} -pattern_type glob -i '{img_path}/*.jpg'  -c:v libx264 -y {plot_path}/{b}.mp4")
            os.system(f'rm -rf {img_path}')

        if trace_plot:
            # trace plot
            start = 15
            length = 20
            frame = np.zeros_like(video[0])
            for i in range(start,start+length):
                current = video[i] * (0.1 + (i-start)/length)
                frame = np.max(np.stack((frame, current)), axis=0)
            cv2.imwrite(f'{plot_path}/{b}_trace.jpg', frame)


def distance_eval_step(gt_pos, pred_pos):
    meds_per_timestep = []
    gt_pred_pairings = None
    for t in range(pred_pos.shape[0]):
        frame_gt = gt_pos[t].cpu().numpy()
        frame_pred = pred_pos[t].cpu().numpy()
        frame_gt = (frame_gt + 1) * 0.5
        frame_pred = (frame_pred + 1) * 0.5

        distances = mm.distances.norm2squared_matrix(frame_gt, frame_pred, max_d2=1)
        if gt_pred_pairings is None:
            frame_gt_ids = list(range(frame_gt.shape[0]))
            frame_pred_ids = list(range(frame_pred.shape[0]))
            gt_pred_pairings = [(frame_gt_ids[g], frame_pred_ids[p]) for g, p in zip(*mm.lap.linear_sum_assignment(distances))]

        med = 0
        for gt_id, pred_id in gt_pred_pairings:
            curr_med = np.sqrt(((frame_gt[gt_id] - frame_pred[pred_id])**2).sum())
            med += curr_med
        if len(gt_pred_pairings) > 0:
            meds_per_timestep.append(med / len(gt_pred_pairings))
        else:
            meds_per_timestep.append(np.nan)
    return meds_per_timestep


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
    evaluation_modes = ['open', 'vidpred_auto', 'vidpred_black_1', 'vidpred_black_2', 'vidpred_black_3', 'vidpred_black_4', 'vidpred_black_5'] # use 'open' for no blackouts
    set_test_array = [set]

    return set_test_array, evaluation_modes
