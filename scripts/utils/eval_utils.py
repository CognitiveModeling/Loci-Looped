import os
import shutil
from einops import rearrange
import torch as th
from model.loci import Loci

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

def compute_position_from_mask(mask):
    """
    Compute the position of the object from the mask.

    Args:
        mask (Tensor[B, N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.
    
    Returns:
        Tensor[B, N, 2]: position of the object
    
    """
    masks_binary = (mask > 0.8).float()[:, :-1]
    b, o, h, w = masks_binary.shape
    masks2 = rearrange(masks_binary, 'b o h w -> (b o) h w')
    boxes = masks_to_boxes(masks2.long())
    boxes = rearrange(boxes, '(b o) c -> b 1 o c', b=b, o=o)
    centroids = boxes_to_centroids(boxes)
    centroids = centroids[:, :, :, [1, 0]].squeeze(1)
    return centroids

def setup_result_folders(file, name, set_test, evaluation_mode, object_view, individual_views):

    net_name = file.split('/')[-1].split('.')[0]
    #root_path = file.split('nets')[0]
    root_path = os.path.join(*file.split('/')[0:-2])
    root_path = os.path.join(root_path, f'results{name}', net_name, set_test['type'])
    plot_path = os.path.join(root_path, evaluation_mode)
    
    # create directories
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)
    os.makedirs(plot_path, exist_ok = True)
    if object_view:
        os.makedirs(os.path.join(plot_path, 'object'), exist_ok = True)
    if individual_views:
        os.makedirs(os.path.join(plot_path, 'individual'), exist_ok = True)
        for group in ['error', 'input', 'background', 'prediction', 'position', 'rawmask', 'mask', 'othermask', 'imagination']:
            os.makedirs(os.path.join(plot_path, 'individual', group), exist_ok = True)
    os.makedirs(os.path.join(root_path, 'statistics'), exist_ok = True)

    # final directory
    plot_path = plot_path + '/'
    print(f"save plots to {plot_path}")

    return root_path, plot_path

def store_statistics(memory, *args, extend=False):
    for i,key in enumerate(memory.keys()):
        if i >= len(args):
            break
        if extend:
            memory[key].extend(args[i])
        else:
            memory[key].append(args[i])
    return memory

def append_statistics(memory1, memory2, ignore=[], extend=False):
    for key in memory1:
        if key not in ignore:
            if extend:
                memory2[key] = memory2[key] + memory1[key]
            else:
                memory2[key].append(memory1[key])
    return memory2

def load_model(cfg, cfg_net, file, device):

    net = Loci(
        cfg_net,
        teacher_forcing    = cfg.defaults.teacher_forcing
    )

    # load model
    if file != '':
        print(f"load {file} to device {device}")
        state = th.load(file, map_location=device)

        # 1. Get keys of current model while ensuring backward compatibility
        model = {}
        allowed_keys = []
        rand_state = net.state_dict()
        for key, value in rand_state.items():
            allowed_keys.append(key)

        # 2. Overwrite with values from file
        for key, value in state["model"].items():
            # replace update_module with percept_gate_controller in key string:
            key = key.replace("update_module", "percept_gate_controller")

            if key in allowed_keys:
                model[key.replace(".module.", ".")] = value
                
        net.load_state_dict(model)

    # ???
    if net.get_init_status() < 1:
        net.inc_init_level()
    
    # set network to evaluation mode
    net = net.to(device=device)

    return net