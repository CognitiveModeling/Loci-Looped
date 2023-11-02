import os
import torch as th
from model.loci import Loci

def setup_result_folders(file, name, set_test, evaluation_mode, object_view, individual_views):

    net_name = file.split('/')[-1].split('.')[0]
    #root_path = file.split('nets')[0]
    root_path = os.path.join(*file.split('/')[0:-1])
    root_path = os.path.join(root_path, f'results{name}', net_name, set_test['type'])
    plot_path = os.path.join(root_path, evaluation_mode)
    
    # create directories
    #if os.path.exists(plot_path):
    #    shutil.rmtree(plot_path)
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

        # backward compatibility
        model = {}
        for key, value in state["model"].items():
            # replace update_module with percept_gate_controller in key string:
            key = key.replace("update_module", "percept_gate_controller")
            model[key.replace(".module.", ".")] = value

        net.load_state_dict(model)

    # ???
    if net.get_init_status() < 1:
        net.inc_init_level()
    
    # set network to evaluation mode
    net = net.to(device=device)

    return net