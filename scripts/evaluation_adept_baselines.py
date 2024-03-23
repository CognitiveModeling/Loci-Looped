from einops import rearrange, reduce, repeat
import torch as th
from torch.utils.data import Dataset, DataLoader, Subset
import cv2
import numpy as np
import pandas as pd
import os
import motmetrics as mm
from scripts.evaluation_adept import calculate_tracking_error, get_evaluation_sets, update_mota_acc
from scripts.utils.eval_utils import boxes_to_centroids, masks_to_boxes, setup_result_folders, store_statistics
from scripts.utils.plot_utils import write_image

FG_THRE = 0.95

def evaluate(dataset: Dataset, file, n, model, plot_frequency= 1, plot_first_samples = 2):

    assert model in ['savi', 'gswm']

    # read pkl file 
    masks_complete = pd.read_pickle(file)

    # plot config
    color_list = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255], [255,255,255]]
    dot_size = 2
    if model == 'savi':
        skip_frames = 2
        offset = 15
    elif model == 'gswm':
        skip_frames = 2
        offset = 0
   
    # memory
    statistics_complete_slots = {'set': [], 'evalmode': [], 'scene': [], 'frame': [], 'slot':[], 'TE': [], 'visible': [], 'bound': [], 'occluder': [], 'inimage': [], 'slot_error': [], 'mask_size': [], 'rawmask_size': [],  'rawmask_size_hidden': [], 'alpha_pos': [], 'alpha_ges': [], 'object_id': []}
    acc_memory_eval = []

    # load adept dataset
    set_test_array, evaluation_modes = get_evaluation_sets(dataset)
    control_samples = set_test_array[0]['samples'] # only consider control set
    evalset = Subset(dataset, control_samples)
    root_path, plot_path = setup_result_folders(file, n, set_test_array[0], evaluation_modes[0], True, False)
    
    for i in range(len(evalset)):
        print(f'Processing sample {i+1}/{len(evalset)}', flush=True)
        input = evalset[i]
        acc = mm.MOTAccumulator(auto_id=True)

        # get input frame and target frame
        tensor = th.tensor(input[0]).float().unsqueeze(0)
        background_fix  = th.tensor(input[1]).unsqueeze(0)
        gt_object_positions = th.tensor(input[3]).unsqueeze(0)
        gt_object_visibility = th.tensor(input[4]).unsqueeze(0)
        gt_occluder_mask = th.tensor(input[5]).unsqueeze(0)

        # apply skip frames
        gt_object_positions = gt_object_positions[:,range(0, tensor.shape[1], skip_frames)]
        gt_object_visibility = gt_object_visibility[:,range(0, tensor.shape[1], skip_frames)]
        tensor = tensor[:,range(0, tensor.shape[1], skip_frames)]
        sequence_len = tensor.shape[1]

        if model == 'savi':
            # load data
            masks = th.tensor(masks_complete['test'][f'control_{i}.mp4']) # N, O, 1, H, W
            masks_before_softmax = th.tensor(masks_complete['test_raw'][f'control_{i}.mp4'])
            imgs_model = None
            recons_model = None

            # calculate rawmasks
            bg_mask = masks_before_softmax.mean(dim=1)
            masks_raw = compute_maskraw(masks_before_softmax, bg_mask, n_slots=7)
            slots_bound = compute_slots_bound(masks_raw)
            
        elif model == 'gswm':
            # load data
            masks = masks_complete[i]['visibility_mask'].squeeze(0)
            masks_raw = masks_complete[i]['object_mask'].squeeze(0)
            slots_bound = masks_complete[i]['z_pres'].squeeze(0)
            slots_bound = (slots_bound > 0.9).float()

            imgs_model = masks_complete[i]['imgs'].squeeze(0)
            imgs_model[:] = imgs_model[:, [2,1,0]]
            recons_model = masks_complete[i]['recon'].squeeze(0)
            recons_model[:] = recons_model[:, [2,1,0]]

            # consider only the first 7 slots
            masks = masks[:,:7]
            masks_raw = masks_raw[:,:7]
            slots_bound = slots_bound[:,:7]
        
        n_slots = masks.shape[1]

        # threshold masks and calculate centroids
        masks_binary = (masks_raw > FG_THRE).float()
        masks2 = rearrange(masks_binary, 't o 1 h w -> (t o) h w')
        boxes = masks_to_boxes(masks2.long())
        boxes = boxes.reshape(1, masks.shape[0], n_slots, 4)
        centroids = boxes_to_centroids(boxes)

        # get rid of batch dimension
        association_table = th.ones(n_slots) * -1

        # iterate over frames
        for t_index in range(offset,min(sequence_len,masks.shape[0])):

            # move to next frame
            input  = tensor[:,t_index]
            gt_positions_target = gt_object_positions[:,t_index]
            gt_visibility_target = gt_object_visibility[:,t_index]

            position_cur = centroids[t_index]
            position_cur = rearrange(position_cur, 'o c -> 1 (o c)')
            slots_bound_cur = slots_bound[t_index]
            slots_bound_cur = rearrange(slots_bound_cur, 'o c -> 1 (o c)')

            # calculate tracking error
            tracking_error, tracking_error_perslot, association_table, slots_visible, slots_in_image, slots_occluder = calculate_tracking_error(gt_positions_target, gt_visibility_target, position_cur, n_slots, slots_bound_cur, None, association_table, gt_occluder_mask)

            rawmask_size = reduce(masks_raw[t_index], 'o 1 h w-> 1 o', 'sum')
            mask_size = reduce(masks[t_index], 'o 1 h w-> 1 o', 'sum')
            
            statistics_complete_slots = store_statistics(statistics_complete_slots,
                                                        ['control'] * n_slots,
                                                        ['control'] * n_slots,
                                                        [control_samples[i]] * n_slots,
                                                        [t_index] * n_slots,
                                                        range(n_slots),
                                                        tracking_error_perslot.cpu().numpy().flatten(),
                                                        slots_visible.cpu().numpy().flatten().astype(int),
                                                        slots_bound_cur.cpu().numpy().flatten().astype(int),
                                                        slots_occluder.cpu().numpy().flatten().astype(int),
                                                        slots_in_image.cpu().numpy().flatten().astype(int),
                                                        [0] * n_slots,
                                                        mask_size.cpu().numpy().flatten(),
                                                        rawmask_size.cpu().numpy().flatten(),
                                                        [0] * n_slots,
                                                        [0] * n_slots,
                                                        [0] * n_slots,
                                                        association_table[0].cpu().numpy().flatten().astype(int),
                                                        extend = True)
            
            acc = update_mota_acc(acc, gt_positions_target, position_cur, slots_bound_cur, n_slots, gt_occluder_mask, slots_occluder, None)

            # plot_option
            if (t_index % plot_frequency == 0) and (i < plot_first_samples) and (t_index >= 0):
                masks_to_display = masks_binary.numpy() # masks_binary.numpy()

                frame = tensor[0, t_index]
                frame = frame.numpy().transpose(1,2,0)
                frame = cv2.resize(frame, (64,64))

                centroids_frame = centroids[t_index]
                centroids_frame[:,0] = (centroids_frame[:,0] + 1) * 64 / 2
                centroids_frame[:,1] = (centroids_frame[:,1] + 1) * 64 / 2

                bound_frame = slots_bound[t_index]
                for c_index,centroid_slot in enumerate(centroids_frame):
                    if bound_frame[c_index] == 1:
                        frame[int(centroid_slot[1]-dot_size):int(centroid_slot[1]+dot_size), int(centroid_slot[0]-dot_size):int(centroid_slot[0]+dot_size)] = color_list[c_index]

                # slot images
                slot_frame = masks_to_display[t_index].max(axis=0)
                slot_frame = slot_frame.reshape((64,64,1)).repeat(3, axis=2)

                if True:
                    for mask in masks_to_display[t_index]:
                        #slot_frame_single = mask.reshape((64,64,1)).repeat(3, axis=2)
                        slot_frame_single = mask.transpose((1,2,0)).repeat(3, axis=2)
                        slot_frame = np.concatenate((slot_frame, slot_frame_single), axis=1)

                if imgs_model is not None:
                    frame_model = imgs_model[t_index].numpy().transpose(1,2,0)
                    recon_model = recons_model[t_index].numpy().transpose(1,2,0)
                    frame = np.concatenate((frame, frame_model, recon_model, slot_frame), axis=1)
                else:
                    frame = np.concatenate((frame, slot_frame), axis=1)
                cv2.imwrite(f'{plot_path}object/objects-{i:04d}-{t_index:03d}.jpg', frame*255)

        acc_memory_eval.append(acc)

    mh = mm.metrics.create()
    summary = mh.compute_many(acc_memory_eval, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    summary['set'] = 'control'
    summary['evalmode'] = 'control'
    pd.DataFrame(summary).to_csv(os.path.join(root_path, 'statistics' , 'accframe.csv'))
    pd.DataFrame(statistics_complete_slots).to_csv(os.path.join(root_path, 'statistics' , 'slotframe.csv'))

def compute_slots_bound(masks):

    # take sum over axis 3,4 with th
    masks_sum = masks.amax(dim=(3,4))
    slots_bound = (masks_sum > FG_THRE).float()
    return slots_bound

def compute_maskraw(mask, bg_mask, n_slots):

    # d is a diagonal matrix which defines what to take the softmax over
    d_mask = th.diag(th.ones(8))
    d_mask[:,-1] = 1
    d_mask[-1,-1] = 0

    mask = mask.squeeze(2)

    # take subset of maskraw with the diagonal matrix
    maskraw = th.cat((mask, bg_mask), dim=1)
    maskraw = repeat(maskraw, 'b o h w -> b r o h w', r = 8)                 
    maskraw = maskraw[:,d_mask.bool()]
    maskraw = rearrange(maskraw, 'b (o r) h w -> b o r h w', o = n_slots)

    # take softmax between each object mask and the background mask
    maskraw = th.squeeze(th.softmax(maskraw, dim=2)[:,:,0], dim=2)
    maskraw = maskraw.unsqueeze(2)

    return maskraw