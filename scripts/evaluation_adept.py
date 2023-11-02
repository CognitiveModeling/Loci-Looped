import torch as th
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn
import os
from scripts.utils.plot_utils import plot_timestep
from scripts.utils.configuration import Configuration
from scripts.utils.io import init_device
import numpy as np
from einops import rearrange, repeat, reduce
import motmetrics as mm
from copy import deepcopy
import pandas as pd
from scripts.utils.eval_utils import append_statistics, load_model, setup_result_folders, store_statistics


def evaluate(cfg: Configuration, dataset: Dataset, file, n, plot_frequency= 1, plot_first_samples = 2):

    # Set up cpu or gpu training
    device, verbose = init_device(cfg)

    # Config
    cfg_net = cfg.model
    cfg_net.batch_size = 1

    # Load model 
    net = load_model(cfg, cfg_net, file, device)
    net.eval()

    # Plot config
    object_view = True
    individual_views = False
    root_path = None

    # get evaluation sets for control and surprise condition
    set_test_array, evaluation_modes = get_evaluation_sets(dataset)

    # memory
    statistics_template = {'set': [], 'evalmode': [], 'scene': [], 'frame': [], 'image_error': [],  'TE': []}
    statistics_complete = deepcopy(statistics_template)
    statistics_complete_slots = {'set': [], 'evalmode': [], 'scene': [], 'frame': [], 'slot':[], 'TE': [], 'visible': [], 'bound': [], 'occluder': [], 'inimage': [], 'slot_error': [], 'mask_size': [], 'rawmask_size': [],  'rawmask_size_hidden': [], 'alpha_pos': [], 'alpha_ges': [], 'object_id': [], 'vanishing': []}
    acc_memory_complete = None  

    for set_test in set_test_array:

        for evaluation_mode in evaluation_modes:
            print(f'Start evaluation loop: {evaluation_mode} - {set_test["type"]}')

            # Load data
            dataloader = DataLoader(
                Subset(dataset, set_test['samples']), 
                num_workers = 1, 
                pin_memory = False, 
                batch_size = 1,
                shuffle = False
            )

            # memory
            mseloss = nn.MSELoss()
            root_path, plot_path = setup_result_folders(file, n, set_test, evaluation_mode, object_view, individual_views)
            acc_memory_eval = []

            with th.no_grad():
                for i, input in enumerate(dataloader):
                    print(f'Processing sample {i+1}/{len(dataloader)}', flush=True)

                    # Data
                    tensor = input[0].float().to(device)
                    background_fix  = input[1].to(device)
                    gt_object_positions = input[3].to(device)
                    gt_object_visibility = input[4].to(device)
                    gt_occluder_mask = input[5].to(device)

                    # Apply skip frames
                    gt_object_positions = gt_object_positions[:,range(0, tensor.shape[1], cfg.defaults.skip_frames)]
                    gt_object_visibility = gt_object_visibility[:,range(0, tensor.shape[1], cfg.defaults.skip_frames)]
                    tensor = tensor[:,range(0, tensor.shape[1], cfg.defaults.skip_frames)]
                    sequence_len = tensor.shape[1]

                    # Placehodlers
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
                    association_table = th.ones(cfg_net.num_objects).to(device) * -1
                    acc = mm.MOTAccumulator(auto_id=True)
                    statistics_batch = deepcopy(statistics_template)
                    slots_vanishing_memory = np.zeros(cfg_net.num_objects)

                    # loop through frames
                    for t_index,t in enumerate(range(-cfg.defaults.teacher_forcing, sequence_len-1)):

                        # Move to next frame
                        t_run = max(t, 0)
                        input  = tensor[:,t_run]
                        target = th.clip(tensor[:,t_run+1], 0, 1)
                        gt_positions_target = gt_object_positions[:,t_run]
                        gt_positions_target_next = gt_object_positions[:,t_run+1]
                        gt_visibility_target = gt_object_visibility[:,t_run]

                        # Forward Pass
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
                            show_hidden = True,
                            clean_slots = (t <= 0),
                        )

                        # 1. Track error
                        if t >= 0:

                            # Position error: MSE between predicted position and target position
                            tracking_error, tracking_error_perslot, association_table, slots_visible, slots_in_image, slots_occluder = calculate_tracking_error(gt_positions_target, gt_visibility_target, position_cur, cfg_net.num_objects, slots_bounded, slots_occluded_cur, association_table, gt_occluder_mask)

                            statistics_batch = store_statistics(statistics_batch,
                                                                set_test['type'],
                                                                evaluation_mode,
                                                                set_test['samples'][i],
                                                                t,
                                                                mseloss(output_next, target).item(),
                                                                tracking_error)

                            # Compute rawmask_size
                            rawmask_size, rawmask_size_hidden, mask_size = compute_mask_sizes(mask_next, rawmask_next, rawmask_hidden)

                            # Compute slot-wise prediction error
                            slot_error = compute_slot_error(cfg_net, target, output_next, mask_next, mask_size)

                            # Check if objects vanishes: they leave the scene suprsingly in the suprrise condition
                            slots_vanishing = compute_vanishing_slots(gt_positions_target, association_table, gt_positions_target_next)
                            slots_vanishing_memory = slots_vanishing + slots_vanishing_memory

                            # Store slot statistics
                            statistics_complete_slots = store_statistics(statistics_complete_slots,
                                                                        [set_test['type']] * cfg_net.num_objects,
                                                                        [evaluation_mode] * cfg_net.num_objects,
                                                                        [set_test['samples'][i]] * cfg_net.num_objects,
                                                                        [t] * cfg_net.num_objects,
                                                                        range(cfg_net.num_objects),
                                                                        tracking_error_perslot.cpu().numpy().flatten(),
                                                                        slots_visible.cpu().numpy().flatten().astype(int),
                                                                        slots_bounded.cpu().numpy().flatten().astype(int),
                                                                        slots_occluder.cpu().numpy().flatten().astype(int),
                                                                        slots_in_image.cpu().numpy().flatten().astype(int),
                                                                        slot_error.cpu().numpy().flatten(),
                                                                        mask_size.cpu().numpy().flatten(),
                                                                        rawmask_size.cpu().numpy().flatten(),
                                                                        rawmask_size_hidden.cpu().numpy().flatten(),
                                                                        slots_closed[:, :, 1].cpu().numpy().flatten(),
                                                                        slots_closed[:, :, 0].cpu().numpy().flatten(),
                                                                        association_table[0].cpu().numpy().flatten().astype(int),
                                                                        extend = True)

                            # Compute MOTA
                            acc = update_mota_acc(acc, gt_positions_target, position_cur, slots_bounded, cfg_net.num_objects, gt_occluder_mask, slots_occluder, rawmask_next)

                        # 2. Remember output
                        mask_last     = mask_next.clone()
                        rawmask_last  = rawmask_next.clone()
                        position_last = position_next.clone()
                        gestalt_last  = gestalt_next.clone()
                        priority_last = priority_next.clone()
                        
                        # 3. Error for next frame
                        bg_error_next = th.sqrt(reduce((target - background)**2, 'b c h w -> b 1 h w', 'mean')).detach()
                        error_next    = th.sqrt(reduce((target - output_next)**2, 'b c h w -> b 1 h w', 'mean')).detach()
                        error_next    = th.sqrt(error_next) * bg_error_next
                        error_last    = error_next.clone()

                        # 4. Plot
                        if (t % plot_frequency == 0) and (i < plot_first_samples) and (t >= 0):
                            plot_timestep(cfg, cfg_net, input, target, mask_cur, mask_next, output_next, position_encoder_cur, position_next, rawmask_hidden, rawmask_cur, rawmask_next, largest_object, object_cur, object_next, object_hidden, slots_bounded, slots_partially_occluded_cur, slots_occluded_cur, slots_partially_occluded_next, slots_occluded_next, slots_closed, gt_positions_target_next, association_table, error_next, output_hidden, object_view, individual_views, statistics_complete_slots, statistics_batch, sequence_len, root_path, plot_path, t_index, t, i)
                                
                    # fill jumping statistics
                    statistics_complete_slots['vanishing'].extend(np.tile(slots_vanishing_memory.astype(int), t+1))

                    # store batch statistics in complete statistics
                    acc_memory_eval.append(acc)
                    statistics_complete = append_statistics(statistics_complete, statistics_batch, extend = True)

            summary = mm.metrics.create().compute_many(acc_memory_eval, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
            summary['set'] = set_test['type']
            summary['evalmode'] = evaluation_mode
            acc_memory_complete = summary.copy() if acc_memory_complete is None else pd.concat([acc_memory_complete, summary])
        
        
    print('-- Evaluation Done --')
    pd.DataFrame(statistics_complete).to_csv(f'{root_path}/statistics/trialframe.csv')
    pd.DataFrame(statistics_complete_slots).to_csv(f'{root_path}/statistics/slotframe.csv')
    pd.DataFrame(acc_memory_complete).to_csv(f'{root_path}/statistics/accframe.csv')
    if object_view and os.path.exists(f'{root_path}/tmp.jpg'):
        os.remove(f'{root_path}/tmp.jpg')
    pass


def compute_vanishing_slots(gt_positions_target, association_table, gt_positions_target_next):
    objects_vanishing = th.abs(gt_positions_target[:,:,2] - gt_positions_target_next[:,:,2]) > 0.2
    objects_vanishing = th.where(objects_vanishing.flatten())[0]
    slots_vanishing = [(obj.item() in objects_vanishing) for obj in association_table[0]]
    return slots_vanishing

def compute_slot_error(cfg_net, target, output_next, mask_next, mask_size):
    output_slot = repeat(mask_next[:,:-1], 'b o h w -> b o 3 h w') * repeat(output_next, 'b c h w -> b o c h w', o=cfg_net.num_objects)
    target_slot = repeat(mask_next[:,:-1], 'b o h w -> b o 3 h w') * repeat(target, 'b c h w -> b o c h w', o=cfg_net.num_objects)
    slot_error = reduce((output_slot - target_slot)**2, 'b o c h w -> b o', 'mean')/mask_size
    return slot_error

def compute_mask_sizes(mask_next, rawmask_next, rawmask_hidden):
    rawmask_size = reduce(rawmask_next[:, :-1], 'b o h w-> b o', 'sum')
    rawmask_size_hidden = reduce(rawmask_hidden[:, :-1], 'b o h w-> b o', 'sum')
    mask_size = reduce(mask_next[:, :-1], 'b o h w-> b o', 'sum')
    return rawmask_size,rawmask_size_hidden,mask_size

def update_mota_acc(acc, gt_positions, estimated_positions, slots_bounded, cfg_num_objects, gt_occluder_mask, slots_occluder, rawmask, ignore_occluder = False):

    # num objects
    num_objects = len(gt_positions[0])

    # get rid of batch dimension and priority dimension
    pos = rearrange(estimated_positions.detach()[0], '(o c) -> o c', o=cfg_num_objects)[:, :2]
    targets = gt_positions[0, :, :2] 

    # stretch positions to account for frame ratio, Specific for ADEPT!
    pos = th.clip(pos, -1, 1)
    pos[:, 0] = pos[:, 0] * 1.5
    targets[:, 0] = targets[:, 0] * 1.5
    
    # remove objects that are not in the image
    edge = 1
    in_image = th.cat([targets[:, 0] < (1.5 * edge), targets[:, 0] > (-1.5 * edge), targets[:, 1] < (1 * edge), targets[:, 1] > (-1 * edge)])
    in_image = th.all(rearrange(in_image, '(c o) -> o c', o=num_objects), dim=1)

    if ignore_occluder:
       in_image = (gt_occluder_mask[0] == 0) * in_image
    targets = targets[in_image]

    # test if position estimates in image
    in_image_pos = th.cat([pos[:, 0] < (1.5 * edge), pos[:, 0] > (-1.5 * edge), pos[:, 1] < (1 * edge), pos[:, 1] > (-1 * edge)])
    in_image_pos = th.all(rearrange(in_image_pos, '(c o) -> c o', o=cfg_num_objects), dim=0, keepdim=True)

    # only position estimates that are in image and bound
    if rawmask is not None:
        rawmask_size = reduce(rawmask[:, :-1], 'b o h w-> b o', 'sum')
        m = (slots_bounded * in_image_pos * (rawmask_size > 100)).bool()
    else:
        m = (slots_bounded * in_image_pos).bool()
    if ignore_occluder:
        m = (m * (1 - slots_occluder)).bool()

    pos = pos[repeat(m, '1 o -> o 2')]
    pos = rearrange(pos, '(o c) -> o c', c = 2)

    # compute pairwise distances
    diagonal_length = th.sqrt(th.sum(th.tensor([2,3])**2)).item()
    C = mm.distances.norm2squared_matrix(targets.cpu().numpy(), pos.cpu().numpy(), max_d2=diagonal_length*0.1)

    # upadate accumulator
    acc.update( (th.where(in_image)[0]).cpu(), (th.where(m)[1]).cpu(), C)

    return acc

def calculate_tracking_error(gt_positions_target, gt_visibility_target, position_cur, cfg_num_slots, slots_bounded, slots_occluded_cur, association_table,  gt_occluder_mask):

    # tracking utils
    pdist = nn.PairwiseDistance(p=2).to(position_cur.device)

    # 1. association of newly bounded slots to ground truth objects
    # num objects
    num_objects = len(gt_positions_target[0])

    # get rid of batch dimension and priority dimension
    pos = rearrange(position_cur.clone()[0], '(o c) -> o c', o=cfg_num_slots)[:, :2]
    targets = gt_positions_target[0, :, :2]

    # stretch positions to account for frame ratio, Specific for ADEPT!
    pos = th.clip(pos, -1, 1)
    pos[:, 0] = pos[:, 0] * 1.5
    targets[:, 0] = targets[:, 0] * 1.5
    diagonal_length = th.sqrt(th.sum(th.tensor([2,3])**2))

    # reshape and repeat for comparison
    pos = repeat(pos, 'o c -> (o r) c', r=num_objects)
    targets = repeat(targets, 'o c -> (r o) c', r=cfg_num_slots)

    # comparison
    distance = pdist(pos, targets)
    distance = rearrange(distance, '(o r) -> o r', r=num_objects)

    # find closest target for each slot
    distance = th.min(distance, dim=1, keepdim=True)

    # update association table
    slots_newly_bounded = slots_bounded * (association_table == -1)
    if slots_occluded_cur is not None:
        slots_newly_bounded = slots_newly_bounded * (1-slots_occluded_cur)
    association_table = association_table * (1-slots_newly_bounded) + slots_newly_bounded * distance[1].T

    # 2. position error
    # get rid of batch dimension and priority dimension
    pos = rearrange(position_cur.clone()[0], '(o c) -> o c', o=cfg_num_slots)[:, :2]
    targets = gt_positions_target[0, :, :3]

    # stretch positions to account for frame ratio, Specific for ADEPT!
    pos[:, 0] = pos[:, 0] * 1.5
    targets[:, 0] = targets[:, 0] * 1.5

    # gather targets according to association table
    targets = targets[association_table.long()][0]

    # determine which slosts are within the image
    slots_in_image = th.cat([targets[:, 0] < 1.5, targets[:, 0] > -1.5, targets[:, 1] < 1, targets[:, 1] > -1, targets[:, 2] > 0])
    slots_in_image = rearrange(slots_in_image, '(c o) -> o c', o=cfg_num_slots)
    slots_in_image = th.all(slots_in_image, dim=1)

    # define which slots to consider for tracking error
    slots_to_track = slots_bounded * slots_in_image

    # compute position error
    targets = targets[:, :2]
    tracking_error_perslot = th.sqrt(th.sum((pos - targets)**2, dim=1))/diagonal_length
    tracking_error_perslot = tracking_error_perslot[None, :] * slots_to_track
    tracking_error = th.sum(tracking_error_perslot).item()/max(th.sum(slots_to_track).item(), 1)
    
    # compute which slots are visible
    visible_objects = th.where(gt_visibility_target[0] == 1)[0]
    slots_visible = th.tensor([[int(obj.item()) in visible_objects for obj in association_table[0]]]).float().to(slots_to_track.device)
    slots_visible = slots_visible * slots_to_track

    # determine which objects are bound to the occluder
    occluder_objects = th.where(gt_occluder_mask[0] == 1)[0]
    slots_occluder = th.tensor([[int(obj.item()) in occluder_objects for obj in association_table[0]]]).float().to(slots_to_track.device)
    slots_occluder = slots_occluder * slots_to_track

    return tracking_error, tracking_error_perslot, association_table, slots_visible, slots_in_image, slots_occluder

def get_evaluation_sets(dataset):

    # Standad evaluation
    evaluation_modes = ['open']
    
    # Important!
    # filter different scenarios: 1 as control and 0,3 as surprise (see Smith et al. 2020)
    suprise_mask = [(sample.case in [1])   for i,sample in enumerate(dataset.samples)]
    control_mask = [(sample.case in [0,3]) for i,sample in enumerate(dataset.samples)]
    
    # Create test sets
    set_surprise = {"samples": np.where(suprise_mask)[0].tolist(), "type": 'surprise'}
    set_control  = {"samples": np.where(control_mask)[0].tolist(), "type": 'control'}
    set_test_array = [set_control, set_surprise]

    return set_test_array, evaluation_modes