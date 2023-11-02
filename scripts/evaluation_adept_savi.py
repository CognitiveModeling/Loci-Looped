from einops import rearrange, reduce, repeat
import torch as th
from torch.utils.data import Dataset, DataLoader, Subset
import cv2
import numpy as np
import pandas as pd
import os
from data.datasets.ADEPT.dataset import AdeptDataset
import motmetrics as mm
from scripts.evaluation_adept import calculate_tracking_error, get_evaluation_sets, update_mota_acc
from scripts.utils.eval_utils import setup_result_folders, store_statistics
from scripts.utils.plot_utils import write_image

FG_THRE = 0.95

def evaluate(dataset: Dataset, file, n, plot_frequency= 1, plot_first_samples = 2):

    # read pkl file 
    masks_complete = pd.read_pickle(file)

    # plot config
    color_list = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255], [255,255,255]]
    dot_size = 2
    skip_frames = 2
    offset = 15
   
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

        # load data
        masks = th.tensor(masks_complete['test'][f'control_{i}.mp4'])
        masks_before_softmax = th.tensor(masks_complete['test_raw'][f'control_{i}.mp4'])

        # calculate rawmasks
        bg_mask = masks_before_softmax.mean(dim=1)
        masks_raw = compute_maskraw(masks_before_softmax, bg_mask)
        slots_bound = compute_slots_bound(masks_raw)

        # threshold masks and calculate centroids
        masks_binary = (masks_raw > FG_THRE).float()
        masks2 = rearrange(masks_binary, 't o 1 h w -> (t o) h w')
        boxes = masks_to_boxes(masks2.long())
        boxes = boxes.reshape(1, masks.shape[0], 7, 4)
        centroids = boxes_to_centroids(boxes)

        # get rid of batch dimension
        association_table = th.ones(7) * -1

        # iterate over frames
        for t_index in range(offset,min(sequence_len,masks.shape[0])):

            # move to next frame
            input  = tensor[:,t_index]
            target = th.clip(tensor[:,t_index+1], 0, 1)
            gt_positions_target = gt_object_positions[:,t_index]
            gt_positions_target_next = gt_object_positions[:,t_index+1]
            gt_visibility_target = gt_object_visibility[:,t_index]

            position_cur = centroids[t_index]
            position_cur = rearrange(position_cur, 'o c -> 1 (o c)')
            slots_bound_cur = slots_bound[t_index]
            slots_bound_cur = rearrange(slots_bound_cur, 'o c -> 1 (o c)')

            # calculate tracking error
            tracking_error, tracking_error_perslot, association_table, slots_visible, slots_in_image, slots_occluder = calculate_tracking_error(gt_positions_target, gt_visibility_target, position_cur, 7, slots_bound_cur, None, association_table, gt_occluder_mask)

            rawmask_size = reduce(masks_raw[t_index], 'o 1 h w-> 1 o', 'sum')
            mask_size = reduce(masks[t_index], 'o 1 h w-> 1 o', 'sum')
            
            statistics_complete_slots = store_statistics(statistics_complete_slots,
                                                        ['control'] * 7,
                                                        ['control'] * 7,
                                                        [control_samples[i]] * 7,
                                                        [t_index] * 7,
                                                        range(7),
                                                        tracking_error_perslot.cpu().numpy().flatten(),
                                                        slots_visible.cpu().numpy().flatten().astype(int),
                                                        slots_bound_cur.cpu().numpy().flatten().astype(int),
                                                        slots_occluder.cpu().numpy().flatten().astype(int),
                                                        slots_in_image.cpu().numpy().flatten().astype(int),
                                                        [0] * 7,
                                                        mask_size.cpu().numpy().flatten(),
                                                        rawmask_size.cpu().numpy().flatten(),
                                                        [0] * 7,
                                                        [0] * 7,
                                                        [0] * 7,
                                                        association_table[0].cpu().numpy().flatten().astype(int),
                                                        extend = True)
            
            acc = update_mota_acc(acc, gt_positions_target, position_cur, slots_bound_cur, 7, gt_occluder_mask, slots_occluder, None)

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

                frame = np.concatenate((frame, slot_frame), axis=1)
                cv2.imwrite(f'{plot_path}object/objects-{i:04d}-{t_index:03d}.jpg', frame*255)

        acc_memory_eval.append(acc)

    mh = mm.metrics.create()
    summary = mh.compute_many(acc_memory_eval, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    summary['set'] = 'control'
    summary['evalmode'] = 'control'
    pd.DataFrame(summary).to_csv(os.path.join(root_path, 'statistics' , 'accframe.csv'))
    pd.DataFrame(statistics_complete_slots).to_csv(os.path.join(root_path, 'statistics' , 'slotframe.csv'))

def masks_to_boxes(masks: th.Tensor) -> th.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return th.zeros((0, 4), device=masks.device, dtype=th.float)

    n = masks.shape[0]

    bounding_boxes = th.zeros((n, 4), device=masks.device, dtype=th.float)

    for index, mask in enumerate(masks):
        if mask.sum() > 0:
            y, x = th.where(mask != 0)

            bounding_boxes[index, 0] = th.min(x)
            bounding_boxes[index, 1] = th.min(y)
            bounding_boxes[index, 2] = th.max(x)
            bounding_boxes[index, 3] = th.max(y)

    return bounding_boxes

def boxes_to_centroids(boxes):
    """Post-process masks instead of directly taking argmax.

    Args:
        bboxes: [B, T, N, 4], 4: [x1, y1, x2, y2]

    Returns:
        centroids: [B, T, N, 2], 2: [x, y]
    """

    centroids = (boxes[:, :, :, :2] + boxes[:, :, :, 2:]) / 2
    centroids = centroids.squeeze(0)

    # scale to [-1, 1]
    centroids[:, :, 0] = centroids[:, :, 0] / 64 * 2 - 1
    centroids[:, :, 1] = centroids[:, :, 1] / 64 * 2 - 1

    return centroids

def compute_slots_bound(masks):

    # take sum over axis 3,4 with th
    masks_sum = masks.amax(dim=(3,4))
    slots_bound = (masks_sum > FG_THRE).float()
    return slots_bound

def compute_maskraw(mask, bg_mask):

    # d is a diagonal matrix which defines what to take the softmax over
    d_mask = th.diag(th.ones(8))
    d_mask[:,-1] = 1
    d_mask[-1,-1] = 0

    mask = mask.squeeze(2)

    # take subset of maskraw with the diagonal matrix
    maskraw = th.cat((mask, bg_mask), dim=1)
    maskraw = repeat(maskraw, 'b o h w -> b r o h w', r = 8)                 
    maskraw = maskraw[:,d_mask.bool()]
    maskraw = rearrange(maskraw, 'b (o r) h w -> b o r h w', o = 7)

    # take softmax between each object mask and the background mask
    maskraw = th.squeeze(th.softmax(maskraw, dim=2)[:,:,0], dim=2)
    maskraw = maskraw.unsqueeze(2)

    return maskraw