import torch as th
from torch import nn
import numpy as np
import cv2
from einops import rearrange, repeat
import matplotlib.pyplot as plt
import PIL
from model.utils.nn_utils import Gaus2D, Vector2D

def preprocess(tensor, scale=1, normalize=False, mean_std_normalize=False):

    if tensor is None:
        return None
    
    if normalize:
        min_ = th.min(tensor)
        max_ = th.max(tensor)
        tensor = (tensor - min_) / (max_ - min_)

    if mean_std_normalize:
        mean = th.mean(tensor)
        std = th.std(tensor)
        tensor = th.clip((tensor - mean) / (2 * std), -1, 1) * 0.5 + 0.5

    if scale > 1:
        upsample = nn.Upsample(scale_factor=scale).to(tensor[0].device)
        tensor = upsample(tensor)

    return tensor

def preprocess_multi(*args, scale):
    return [preprocess(a, scale) for a in args]

def color_mask(mask):

    colors = th.tensor([
	[ 255,   0,   0 ],
	[   0,   0, 255 ],
	[ 255, 255,   0 ],
	[ 255,   0, 255 ],
	[   0, 255, 255 ],
	[   0, 255,   0 ],
	[ 255, 128,   0 ],
	[ 128, 255,   0 ],
	[ 128,   0, 255 ],
	[ 255,   0, 128 ],
	[   0, 255, 128 ],
	[   0, 128, 255 ],
	[ 255, 128, 128 ],
	[ 128, 255, 128 ],
	[ 128, 128, 255 ],
	[ 255, 128, 128 ],
	[ 128, 255, 128 ],
	[ 128, 128, 255 ],
	[ 255, 128, 255 ],
	[ 128, 255, 255 ],
	[ 128, 255, 255 ],
	[ 255, 255, 128 ],
	[ 255, 255, 128 ],
	[ 255, 128, 255 ],
	[ 128,   0,   0 ],
	[   0,   0, 128 ],
	[ 128, 128,   0 ],
	[ 128,   0, 128 ],
	[   0, 128, 128 ],
	[   0, 128,   0 ],
	[ 128, 128,   0 ],
	[ 128, 128,   0 ],
	[ 128,   0, 128 ],
	[ 128,   0, 128 ],
	[   0, 128, 128 ],
	[   0, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
    ], device = mask.device) / 255.0

    colors = colors.view(1, -1, 3, 1, 1)
    mask = mask.unsqueeze(dim=2)

    return th.sum(colors[:,:mask.shape[1]] * mask, dim=1)

def get_color(o):
    colors = th.tensor([
	[ 255,   0,   0 ],
	[   0,   0, 255 ],
	[ 255, 255,   0 ],
	[ 255,   0, 255 ],
	[   0, 255, 255 ],
	[   0, 255,   0 ],
	[ 255, 128,   0 ],
	[ 128, 255,   0 ],
	[ 128,   0, 255 ],
	[ 255,   0, 128 ],
	[   0, 255, 128 ],
	[   0, 128, 255 ],
	[ 255, 128, 128 ],
	[ 128, 255, 128 ],
	[ 128, 128, 255 ],
	[ 255, 128, 128 ],
	[ 128, 255, 128 ],
	[ 128, 128, 255 ],
	[ 255, 128, 255 ],
	[ 128, 255, 255 ],
	[ 128, 255, 255 ],
	[ 255, 255, 128 ],
	[ 255, 255, 128 ],
	[ 255, 128, 255 ],
	[ 128,   0,   0 ],
	[   0,   0, 128 ],
	[ 128, 128,   0 ],
	[ 128,   0, 128 ],
	[   0, 128, 128 ],
	[   0, 128,   0 ],
	[ 128, 128,   0 ],
	[ 128, 128,   0 ],
	[ 128,   0, 128 ],
	[ 128,   0, 128 ],
	[   0, 128, 128 ],
	[   0, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
    ]) / 255.0

    colors = colors.view(48,3)
    return colors[o]

def to_rgb(tensor: th.Tensor):
    return th.cat((
        tensor * 0.6 + 0.4,
        tensor, 
        tensor
    ), dim=1)

def visualise_gate(gate, h, w, invert = False):
    bar = th.ones((1,h,w), device=gate.device) * 0.9
    black = int(w*gate.item())
    black = w-black if invert else black
    if black > 0:
        bar[:,:, -black:] = 0
    return bar

def get_highlighted_input(input, mask_cur):

    # highlight error
    highlighted_input = input
    if mask_cur is not None:
        grayscale        = input[:,0:1] * 0.299 + input[:,1:2] * 0.587 + input[:,2:3] * 0.114
        object_mask_cur  = th.sum(mask_cur[:,:-1], dim=1).unsqueeze(dim=1)
        highlighted_input  = grayscale * (1 - object_mask_cur) 
        highlighted_input += grayscale * object_mask_cur * 0.3333333 
        cmask = color_mask(mask_cur[:,:-1])
        highlighted_input  = highlighted_input + cmask * 0.6666666

    return highlighted_input

def color_slots(image, slots_bounded, slots_partially_occluded_cur, slots_occluded_cur):

    image = (1-image) * slots_bounded + image * (1-slots_bounded)
    image = th.clip(image - 0.3, 0,1) * slots_partially_occluded_cur + image * (1-slots_partially_occluded_cur)
    image = th.clip(image - 0.3, 0,1) * slots_occluded_cur + image * (1-slots_occluded_cur)

    return image

def compute_occlusion_mask(rawmask_cur, rawmask_next,  mask_cur, mask_next, scale):

    # compute occlusion mask
    occluded_cur    = th.clip(rawmask_cur - mask_cur, 0, 1)[:,:-1]
    occluded_next   = th.clip(rawmask_next - mask_next, 0, 1)[:,:-1]

    # to rgb
    rawmask_cur     = repeat(rawmask_cur[:,:-1], 'b o h w -> b (o 3) h w')
    rawmask_next    = repeat(rawmask_next[:,:-1], 'b o h w -> b (o 3) h w')

    # scale 
    occluded_next   = preprocess(occluded_next, scale)
    occluded_cur    = preprocess(occluded_cur, scale)
    rawmask_cur     = preprocess(rawmask_cur, scale)
    rawmask_next    = preprocess(rawmask_next, scale)

    # set occlusion to red
    rawmask_cur         = rearrange(rawmask_cur, 'b (o c) h w -> b o c h w', c = 3)
    rawmask_cur[:,:,0]  = rawmask_cur[:,:,0] * (1 - occluded_next)
    rawmask_cur[:,:,1]  = rawmask_cur[:,:,1] * (1 - occluded_next)

    rawmask_next        = rearrange(rawmask_next, 'b (o c) h w -> b o c h w', c = 3)
    rawmask_next[:,:,0] = rawmask_next[:,:,0] * (1 - occluded_next)
    rawmask_next[:,:,1] = rawmask_next[:,:,1] * (1 - occluded_next)

    return rawmask_cur, rawmask_next

def plot_online_error_slots(errors, error_name, target, sequence_len, root_path, visibilty_memory, slots_bounded, ylim=0.3):
    error_plots = []
    if len(errors) > 0:
        num_slots = int(th.sum(slots_bounded).item())
        errors = rearrange(np.array(errors), '(l o) -> o l', o=len(slots_bounded))[:num_slots]
        visibilty_memory = rearrange(np.array(visibilty_memory), '(l o) -> o l', o=len(slots_bounded))[:num_slots]
        for error,visibility in zip(errors, visibilty_memory):

            if len(error) < sequence_len:
                fig, ax = plt.subplots(figsize=(round(target.shape[3]/100,2), round(target.shape[2]/100,2)))
                plt.plot(error, label=error_name)

                visibility = np.concatenate((visibility, np.ones(sequence_len-len(error))))
                ax.fill_between(range(sequence_len), 0, 1, where=visibility==0, color='orange', alpha=0.3, transform=ax.get_xaxis_transform())
                
                plt.xlim((0,sequence_len))
                plt.ylim((0,ylim))
                fig.tight_layout()
                plt.savefig(f'{root_path}/tmp.jpg')

                error_plot = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
                error_plot = th.from_numpy(np.array(error_plot).transpose(2,0,1))
                plt.close(fig)  

                error_plots.append(error_plot)

    return error_plots

def plot_online_error(error, error_name, target, t, i, sequence_len, root_path, online_surprise = False):

    fig = plt.figure(figsize=( round(target.shape[3]/50,2), round(target.shape[2]/50,2) ))
    plt.plot(error, label=error_name)
    
    if online_surprise:
        # compute moving average of error
        moving_average_length = 10
        if t > moving_average_length:
            moving_average_length += 1
            average_error = np.mean(error[-moving_average_length:-1])
            current_sd = np.std(error[-moving_average_length:-1])
            current_error = error[-1]

            if current_error > average_error + 2 * current_sd:
                fig.set_facecolor('orange')

    plt.xlim((0,sequence_len))
    plt.legend()
    # increase title size
    plt.title(f'{error_name}', fontsize=20)
    plt.xlabel('timestep')
    plt.ylabel('error')
    plt.savefig(f'{root_path}/tmp.jpg')

    error_plot = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    error_plot = th.from_numpy(np.array(error_plot).transpose(2,0,1))
    plt.close(fig)  

    return error_plot

def plot_object_view(error_plot, error_plot2, error_plot_slots, error_plot_slots2, highlighted_input, output_hidden, object_next, rawmask_next, velocity_next2d, target, slots_closed, gt_positions_target_next, association_table, size, num_objects, largest_object, rollout_mode=False, openings=None):

    # add ground truth positions of objects to image
    target = target.clone()
    if gt_positions_target_next is not None:
        for o in range(gt_positions_target_next.shape[1]):
            position = gt_positions_target_next[0, o]
            position = position/2 + 0.5

            if (len(position.shape) < 3 or position[2] > 0.0) and position[0] > 0.0 and position[0] < 1.0 and position[1] > 0.0 and position[1] < 1.0:
                width = int(target.shape[2]*0.05)
                w = np.clip(int(position[0]*target.shape[2]), width, target.shape[2]-width).item() # made for bouncing balls
                h = np.clip(int(position[1]*target.shape[3]), width, target.shape[3]-width).item()
                col = get_color(o).view(3,1,1)
                target[0,:,(w-width):(w+width), (h-width):(h+width)] = col

                # add these positions to the associated slots velocity_next2d ilustration
                if association_table is not None:
                    slots = (association_table[0] == o).nonzero()
                    for s in slots.flatten():
                        velocity_next2d[s,:,(w-width):(w+width), (h-width):(h+width)] = col

                        if output_hidden is not None and s != largest_object:
                            output_hidden[0,:,(w-width):(w+width), (h-width):(h+width)] = col

    gateheight = 60
    ch = 40
    gh = 40
    gh_bar = gh-20
    gh_margin = int((gh-gh_bar)/2)
    margin = 20
    slots_margin  = 10
    height = size[0] * 6 + 18*6
    width  = size[1] * 4 + 18*2 + size[1]*num_objects + 6*(num_objects+1) + slots_margin*(num_objects+1)
    img = th.ones((3, height, width), device = object_next.device) * 0.4
    row = (lambda row_index: [2*size[0]*row_index + (row_index+1)*margin,   2*size[0]*(row_index+1) + (row_index+1)*margin])
    col1 = range(margin, margin + size[1]*2)
    col2 = range(width-(margin+size[1]*2), width-margin)

    # add frame around image
    if rollout_mode:
        img[0,margin-2:margin+size[0]*2+2, margin-2:margin+size[1]*2+2] = 1

    img[:,row(0)[0]:row(0)[1], col1] = preprocess(highlighted_input.to(object_next.device), 2)[0] 
    img[:,row(1)[0]:row(1)[1], col1] = preprocess(output_hidden.to(object_next.device), 2)[0]
    img[:,row(2)[0]:row(2)[1], col1] = preprocess(target.to(object_next.device), 2)[0] 

    # add large error plots to image
    if error_plot is not None:
        img[:,row(0)[0]+gh+ch+2*margin-gh_margin:row(0)[1]+gh+ch+2*margin-gh_margin, col2] = preprocess(error_plot.to(object_next.device), normalize= True)
    if error_plot2 is not None:
        img[:,row(2)[0]:row(2)[1], col2] = preprocess(error_plot2.to(object_next.device), normalize= True)

    # fill colunmns with slots
    for o in range(num_objects):

        col = 18+size[1]*2+6+o*(6+size[1])+(o+1)*slots_margin
        col = range(col, col + size[1])

        # color bar for the gate
        if (error_plot_slots2 is not None) and len(error_plot_slots2) > o:
            img[:,margin:margin+ch,  col] = get_color(o).view(3,1,1).to(object_next.device)

        # gestalt gate
        img[:,margin+ch+2*margin:2*margin+gh_bar+ch+margin,  col] = visualise_gate(slots_closed[:,o, 0].to(object_next.device), h=gh_bar, w=len(col))
        offset = gh+margin-gh_margin+ch+2*margin
        row = (lambda row_index: [offset+(size[0]+6)*row_index, offset+size[0]*(row_index+1)+6*row_index])
        
        img[:,row(0)[0]:row(0)[1],  col] = preprocess(rawmask_next[0,o].to(object_next.device))
        img[:,row(1)[0]:row(1)[1],  col] = preprocess(object_next[:,o].to(object_next.device))

        # small error plots top row
        if (error_plot_slots2 is not None) and len(error_plot_slots2) > o:
            img[:,row(2)[0]:row(2)[1],  col] = preprocess(error_plot_slots2[o].to(object_next.device), normalize=True)

        # switch to bottom row
        offset = margin*2-8
        row = (lambda row_index: [offset+(size[0]+6)*row_index, offset+size[0]*(row_index+1)+6*row_index])   

        # position gate
        img[:,row(4)[0]-gh+gh_margin:row(4)[0]-gh_margin,  col] = visualise_gate(slots_closed[:,o, 1].to(object_next.device), h=gh_bar, w=len(col))
        img[:,row(4)[0]:row(4)[1],  col]    = preprocess(velocity_next2d[o].to(object_next.device), normalize=True)[0]

        # small error plots bottom row
        if (error_plot_slots is not None) and len(error_plot_slots) > o:
            img[:,row(5)[0]:row(5)[1],  col] = preprocess(error_plot_slots[o].to(object_next.device), normalize=True)

        # add gatelord gate visualisation to image
        if openings is not None:
            img[:,row(5)[1]+gh_margin:row(5)[1]+gh-gh_margin,  col] = visualise_gate(openings[:,o].to(object_next.device), h=gh_bar, w=len(col), invert = True)

    img = rearrange(img * 255, 'c h w -> h w c').cpu()

    return img

def write_image(file, img):
    img = rearrange(img * 255, 'c h w -> h w c').cpu().numpy()
    cv2.imwrite(file, img)

    pass

def extract_element(tensor, index):
    if tensor is None:
        return None
    return tensor[index:index+1]

def plot_timestep(cfg, cfg_net, input, target, mask_cur, mask_next, output_next, position_encoder_cur, position_next, rawmask_hidden, rawmask_cur, rawmask_next, largest_object, object_cur, object_next, object_hidden, slots_bounded, slots_partially_occluded_cur, slots_occluded_cur, slots_partially_occluded_next, slots_occluded_next, slots_closed, gt_positions_target_next, association_table, error_next, output_hidden, object_view, individual_views, statistics_complete_slots, statistics_batch, sequence_len, root_path, plot_path, t_index, t, sample_i, rollout_mode=False,  num_vid=2, att=None, openings=None):

    if len(input) > 1:
        img_list = None
        for i in range(min(len(input), num_vid)):
            _input = extract_element(input, i)
            _target = extract_element(target, i)
            _mask_cur = extract_element(mask_cur, i)
            _mask_next = extract_element(mask_next, i)
            _output_next = extract_element(output_next, i)
            _position_encoder_cur = extract_element(position_encoder_cur, i)
            _position_next = extract_element(position_next, i)
            _rawmask_hidden = extract_element(rawmask_hidden, i)
            _rawmask_cur = extract_element(rawmask_cur, i)
            _rawmask_next = extract_element(rawmask_next, i)
            _largest_object = extract_element(largest_object, i)
            _object_cur = extract_element(object_cur, i)
            _object_next = extract_element(object_next, i)
            _object_hidden = extract_element(object_hidden, i)
            _slots_bounded = extract_element(slots_bounded, i)
            _slots_partially_occluded_cur = extract_element(slots_partially_occluded_cur, i)
            _slots_occluded_cur = extract_element(slots_occluded_cur, i)
            _slots_partially_occluded_next = extract_element(slots_partially_occluded_next, i)
            _slots_occluded_next = extract_element(slots_occluded_next, i)
            _slots_closed = extract_element(slots_closed, i)
            _gt_positions_target_next = extract_element(gt_positions_target_next, i)
            _association_table = extract_element(association_table, i)
            _error_next = extract_element(error_next, i)
            _output_hidden = extract_element(output_hidden, i)
            _att = extract_element(att, i)
            _openings = extract_element(openings, i)

            img = plot_timestep_single(cfg, cfg_net, _input, _target, _mask_cur, _mask_next, _output_next, _position_encoder_cur, _position_next, _rawmask_hidden, _rawmask_cur, _rawmask_next, _largest_object, _object_cur, _object_next, _object_hidden, _slots_bounded, _slots_partially_occluded_cur, _slots_occluded_cur, _slots_partially_occluded_next, _slots_occluded_next, _slots_closed, _gt_positions_target_next, _association_table, _error_next, _output_hidden, object_view, individual_views, statistics_complete_slots, statistics_batch, sequence_len, root_path, plot_path, t_index, t, i, rollout_mode, att=_att, openings=_openings)
            if img_list is None:
                img_list = img.unsqueeze(0)
            else:
                img_list = th.cat((img_list, img.unsqueeze(0)), dim=0)

        return img_list.permute(0, 3, 1, 2)
    
    else:
        img = plot_timestep_single(cfg, cfg_net, input, target, mask_cur, mask_next, output_next, position_encoder_cur, position_next, rawmask_hidden, rawmask_cur, rawmask_next, largest_object, object_cur, object_next, object_hidden, slots_bounded, slots_partially_occluded_cur, slots_occluded_cur, slots_partially_occluded_next, slots_occluded_next, slots_closed, gt_positions_target_next, association_table, error_next, output_hidden, object_view, individual_views, statistics_complete_slots, statistics_batch, sequence_len, root_path, plot_path, t_index, t, sample_i, rollout_mode, att=att, openings=openings)
        return img

def plot_timestep_single(cfg, cfg_net, input, target, mask_cur, mask_next, output_next, position_encoder_cur, position_next, rawmask_hidden, rawmask_cur, rawmask_next, largest_object, object_cur, object_next, object_hidden, slots_bounded, slots_partially_occluded_cur, slots_occluded_cur, slots_partially_occluded_next, slots_occluded_next, slots_closed, gt_positions_target_next, association_table, error_next, output_hidden, object_view, individual_views, statistics_complete_slots, statistics_batch, sequence_len, root_path, plot_path, t_index, t, i, rollout_mode=False, att=None, openings=None):

    # Create eposition helpers
    size, gaus2d, vector2d, scale = get_position_helper(cfg_net, mask_cur.device)
    
    # Compute plot content
    highlighted_input = get_highlighted_input(input, mask_cur)
    output = th.clip(output_next, 0, 1)
    position_cur2d = gaus2d(rearrange(position_encoder_cur, 'b (o c) -> (b o) c', o=cfg_net.num_objects))
    velocity_next2d = vector2d(rearrange(position_next, 'b (o c) -> (b o) c', o=cfg_net.num_objects))      

    # color slots
    slots_bounded, slots_partially_occluded_cur, slots_occluded_cur, slots_partially_occluded_next, slots_occluded_next = reshape_slots(slots_bounded, slots_partially_occluded_cur, slots_occluded_cur, slots_partially_occluded_next, slots_occluded_next)
    position_cur2d = color_slots(position_cur2d, slots_bounded, slots_partially_occluded_cur, slots_occluded_cur)
    velocity_next2d = color_slots(velocity_next2d, slots_bounded, slots_partially_occluded_next, slots_occluded_next)

    # compute occlusion
    if (cfg.datatype == "adept") and rawmask_hidden is not None:
        rawmask_cur_l, rawmask_next_l = compute_occlusion_mask(rawmask_cur, rawmask_next, mask_cur, mask_next, scale)
        rawmask_cur_h, rawmask_next_h = compute_occlusion_mask(rawmask_cur, rawmask_hidden, mask_cur, mask_next, scale)
        rawmask_cur_h[:,largest_object] = rawmask_cur_l[:,largest_object]
        rawmask_next_h[:,largest_object] = rawmask_next_l[:,largest_object]
        rawmask_cur = rawmask_cur_h
        rawmask_next = rawmask_next_h
        object_hidden[:, largest_object] = object_next[:, largest_object]
        object_next = object_hidden
    else:
        rawmask_cur, rawmask_next = compute_occlusion_mask(rawmask_cur, rawmask_next, mask_cur, mask_next, scale)

    # scale plot content
    input, target, output, highlighted_input, object_next, object_cur, mask_next, error_next, output_hidden, output_next = preprocess_multi(input, target, output, highlighted_input, object_next, object_cur, mask_next, error_next, output_hidden, output_next, scale=scale)

    # reshape
    object_next     = rearrange(object_next, 'b (o c) h w -> b o c h w', c = cfg_net.img_channels)
    object_cur      = rearrange(object_cur, 'b (o c) h w -> b o c h w', c = cfg_net.img_channels)
    mask_next       = rearrange(mask_next, 'b (o 1) h w -> b o 1 h w')

    if object_view:
        if (cfg.datatype == "adept") and statistics_complete_slots is not None:
            num_objects = 4
            error_plot_slots = plot_online_error_slots(statistics_complete_slots['TE'][-cfg_net.num_objects*(t+1):], 'Tracking error', target, sequence_len, root_path, statistics_complete_slots['visible'][-cfg_net.num_objects*(t+1):], slots_bounded)
            #error_plot_slots2 = plot_online_error_slots(statistics_complete_slots['slot_error'][-cfg_net.num_objects*(t+1):], 'Image error', target, sequence_len, root_path, statistics_complete_slots['visible'][-cfg_net.num_objects*(t+1):], slots_bounded, ylim=0.0001)
            error_plot = plot_online_error(statistics_batch['image_error'], 'Prediction error', target, t, i, sequence_len, root_path)
            error_plot2 = plot_online_error(statistics_batch['TE'], 'Tracking error', target, t, i, sequence_len, root_path)
            att_histogram = plot_attention_histogram(att, target, root_path)
            img = plot_object_view(error_plot, error_plot2, error_plot_slots, att_histogram, highlighted_input, output_hidden, object_next, rawmask_next, velocity_next2d, target, slots_closed, gt_positions_target_next, association_table, size, num_objects, largest_object, openings=openings)
        elif (cfg.datatype == "clevrer") and statistics_complete_slots is not None:
            num_objects = cfg_net.num_objects
            error_plot_slots2 = plot_online_error_slots(statistics_complete_slots['slot_error'][-cfg_net.num_objects*(t+1):], 'Image error', target, sequence_len, root_path, statistics_complete_slots['slot_error'][-cfg_net.num_objects*(t+1):], slots_bounded, ylim=0.0001)
            error_plot = plot_online_error(statistics_batch['image_error_mse'], 'Prediction error', target, t, i, sequence_len, root_path)
            img = plot_object_view(error_plot, None, None, error_plot_slots2, highlighted_input, output_next, object_next, rawmask_next, velocity_next2d, target, slots_closed, gt_positions_target_next, association_table, size, num_objects, largest_object, openings=openings)
        else:
            num_objects = cfg_net.num_objects
            att_histogram = plot_attention_histogram(att, target, root_path)
            img = plot_object_view(None, None, att_histogram, None, input, output, object_next, rawmask_next, velocity_next2d, target, slots_closed, gt_positions_target_next, association_table, size, num_objects, largest_object, rollout_mode=rollout_mode, openings=openings)

        if plot_path is not None:
            cv2.imwrite(f'{plot_path}/object/{i:04d}-{t_index:03d}.jpg', img.numpy())

    if individual_views:
        # ['error', 'input', 'background', 'prediction', 'position', 'rawmask', 'mask', 'othermask']:
        write_image(f'{plot_path}/individual/error/error-{i:04d}-{t_index:03d}.jpg', error_next[0])
        write_image(f'{plot_path}/individual/input/input-{i:04d}-{t_index:03d}.jpg', target[0])
        write_image(f'{plot_path}/individual/background/background-{i:04d}-{t_index:03d}.jpg', mask_next[0,-1])
        #write_image(f'{plot_path}/individual/imagination/imagination-{i:04d}-{t_index:03d}.jpg', output_hidden[0])
        write_image(f'{plot_path}/individual/prediction/prediction-{i:04d}-{t_index:03d}.jpg', output_next[0])
        for o in range(len(rawmask_next[0])):
            write_image(f'{plot_path}/individual/rgb/object-{i:04d}-{o}-{t_index:03d}.jpg', object_next[0][o])
            write_image(f'{plot_path}/individual/rawmask/rawmask-{i:04d}-{o}-{t_index:03d}.jpg', rawmask_next[0][o])

    return img

def get_position_helper(cfg_net, device):
    size = cfg_net.input_size
    gaus2d  = Gaus2D(size).to(device)
    vector2d = Vector2D(size).to(device)
    scale  = size[0] // (cfg_net.latent_size[0] * 2**(cfg_net.level*2))
    return size,gaus2d,vector2d,scale

def reshape_slots(slots_bounded, slots_partially_occluded_cur, slots_occluded_cur, slots_partially_occluded_next, slots_occluded_next):

    slots_bounded = th.squeeze(slots_bounded)[..., None,None,None]
    slots_partially_occluded_cur = th.squeeze(slots_partially_occluded_cur)[..., None,None,None]
    slots_occluded_cur = th.squeeze(slots_occluded_cur)[..., None,None,None]
    slots_partially_occluded_next = th.squeeze(slots_partially_occluded_next)[..., None,None,None]
    slots_occluded_next = th.squeeze(slots_occluded_next)[..., None,None,None]

    return slots_bounded, slots_partially_occluded_cur, slots_occluded_cur, slots_partially_occluded_next, slots_occluded_next

def plot_attention_histogram(att, target, root_path):
    att_plots = []
    if (att is not None) and (len(att) > 0):
        att = att[0]
        for object_attention in att:

            fig, ax = plt.subplots(figsize=(round(target.shape[3]/100,2), round(target.shape[2]/100,2)))

            # Plot a bar plot over the 6 objects
            num_objects = len(object_attention)
            ax.bar(range(num_objects), object_attention.cpu())
            ax.set_ylim([0,1])
            ax.set_xlim([-1,num_objects])
            ax.set_xticks(range(num_objects))
            #ax.set_xticklabels(['1','2','3','4','5','6'])
            ax.set_ylabel('attention')
            ax.set_xlabel('object')
            ax.set_title('Attention histogram')

            # fixed
            fig.tight_layout()
            plt.savefig(f'{root_path}/tmp.jpg')
            plot = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
            plot = th.from_numpy(np.array(plot).transpose(2,0,1))
            plt.close(fig)  
            att_plots.append(plot)

        return att_plots
    else:
        return None