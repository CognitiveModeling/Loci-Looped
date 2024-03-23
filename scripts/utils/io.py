import os
from scripts.utils.configuration import Configuration
import time
import torch as th
import numpy as np
from einops import rearrange, repeat, reduce


def init_device(cfg):
    print(f'Cuda available: {th.cuda.is_available()} Cuda count: {th.cuda.device_count()}')
    if th.cuda.is_available():
        device = th.device("cuda:0")
        verbose = False
        cfg.device = "cuda:0"
        cfg.model.device = "cuda:0"
        print('!!! USING GPU !!!')
    else:
        device = th.device("cpu")
        verbose = True
        cfg.device = "cpu"
        cfg.model.device = "cpu"
        cfg.model.batch_size = 2
        cfg.defaults.teacher_forcing = 4
        print('!!! USING CPU !!!')
    return device,verbose

class Timer:
    
    def __init__(self):
        self.last   = time.time()
        self.passed = 0
        self.sum    = 0

    def __str__(self):
        self.passed = self.passed * 0.99 + time.time() - self.last
        self.sum    = self.sum * 0.99 + 1
        passed      = self.passed / self.sum
        self.last = time.time()

        if passed > 1:
            return f"{passed:.2f}s/it"

        return f"{1.0/passed:.2f}it/s"

class UEMA:
    
    def __init__(self, memory = 100):
        self.value  = 0
        self.sum    = 1e-30
        self.decay  = np.exp(-1 / memory)

    def update(self, value):
        self.value = self.value * self.decay + value
        self.sum   = self.sum   * self.decay + 1

    def __float__(self):
        return self.value / self.sum


def model_path(cfg: Configuration, overwrite=False, move_old=True):
    """
    Makes the model path, option to not overwrite
    :param cfg: Configuration file with the model path
    :param overwrite: Overwrites the files in the directory, else makes a new directory
    :param move_old: Moves old folder with the same name to an old folder, if not overwrite
    :return: Model path
    """
    _path = os.path.join('out', cfg.dataset)
    path = os.path.join(_path, cfg.model_path)

    if not os.path.exists(_path):
        os.makedirs(_path)

    if not overwrite:
        if move_old:
            # Moves existing directory to an old folder
            if os.path.exists(path):
                old_path = os.path.join(_path, f'{cfg.model_path}_old')
                if not os.path.exists(old_path):
                    os.makedirs(old_path)
                _old_path = os.path.join(old_path, cfg.model_path)
                i = 0
                while os.path.exists(_old_path):
                    i = i + 1
                    _old_path = os.path.join(old_path, f'{cfg.model_path}_{i}')
                os.renames(path, _old_path)
        else:
            # Increases number after directory name for each new path
            i = 0
            while os.path.exists(path):
                i = i + 1
                path = os.path.join(_path, f'{cfg.model_path}_{i}')

    return path

class LossLogger:

    def __init__(self, writer):

        self.avgloss                  = UEMA()
        self.avg_position_loss        = UEMA()
        self.avg_time_loss            = UEMA()
        self.avg_latent_loss          = UEMA()
        self.avg_encoding_loss        = UEMA()
        self.avg_prediction_loss      = UEMA()
        self.avg_prediction_loss_long = UEMA(33333)
        self.avg_num_objects          = UEMA()
        self.avg_openings             = UEMA()
        self.avg_gestalt              = UEMA()
        self.avg_gestalt2             = UEMA()
        self.avg_gestalt_mean         = UEMA()
        self.avg_update_gestalt       = UEMA()
        self.avg_update_position      = UEMA()
        self.avg_num_bounded          = UEMA()

        self.writer = writer

    
    def update_complete(self, avg_position_loss, avg_time_loss, avg_latent_loss, avg_encoding_loss, avg_prediction_loss, avg_num_objects, avg_openings, avg_gestalt, avg_gestalt2, avg_gestalt_mean, avg_update_gestalt, avg_update_position, avg_num_bounded, lr, num_updates):

        self.avg_position_loss.update(avg_position_loss.item())
        self.avg_time_loss.update(avg_time_loss.item())
        self.avg_latent_loss.update(avg_latent_loss.item())
        self.avg_encoding_loss.update(avg_encoding_loss.item())
        self.avg_prediction_loss.update(avg_prediction_loss.item())
        self.avg_prediction_loss_long.update(avg_prediction_loss.item())
        self.avg_num_objects.update(avg_num_objects)
        self.avg_openings.update(avg_openings)
        self.avg_gestalt.update(avg_gestalt.item())
        self.avg_gestalt2.update(avg_gestalt2.item())
        self.avg_gestalt_mean.update(avg_gestalt_mean.item())
        self.avg_update_gestalt.update(avg_update_gestalt)
        self.avg_update_position.update(avg_update_position)
        self.avg_num_bounded.update(avg_num_bounded)

        self.writer.add_scalar("Train/Position Loss", avg_position_loss.item(), num_updates)
        self.writer.add_scalar("Train/Time Loss", avg_time_loss.item(), num_updates)
        self.writer.add_scalar("Train/Latent Loss", avg_latent_loss.item(), num_updates)
        self.writer.add_scalar("Train/Encoder Loss", avg_encoding_loss.item(), num_updates)
        self.writer.add_scalar("Train/Prediction Loss", avg_prediction_loss.item(), num_updates)
        self.writer.add_scalar("Train/Number of Objects", avg_num_objects, num_updates)
        self.writer.add_scalar("Train/Openings", avg_openings, num_updates)
        self.writer.add_scalar("Train/Gestalt", avg_gestalt.item(), num_updates)
        self.writer.add_scalar("Train/Gestalt2", avg_gestalt2.item(), num_updates)
        self.writer.add_scalar("Train/Gestalt Mean", avg_gestalt_mean.item(), num_updates)
        self.writer.add_scalar("Train/Update Gestalt", avg_update_gestalt, num_updates)
        self.writer.add_scalar("Train/Update Position", avg_update_position, num_updates)
        self.writer.add_scalar("Train/Number Bounded", avg_num_bounded, num_updates)
        self.writer.add_scalar("Train/Learning Rate", lr, num_updates)

        pass

    def update_average_loss(self, avgloss, num_updates):
        self.avgloss.update(avgloss)
        self.writer.add_scalar("Train/Loss", avgloss, num_updates)
        pass

    def get_log(self):
        info = f'Loss: {np.abs(float(self.avgloss)):.2e}|{float(self.avg_prediction_loss):.2e}|{float(self.avg_prediction_loss_long):.2e}, reg: {float(self.avg_encoding_loss):.2e}|{float(self.avg_time_loss):.2e}|{float(self.avg_latent_loss):.2e}|{float(self.avg_position_loss):.2e}, obj: {float(self.avg_num_objects):.1f}, open: {float(self.avg_openings):.2e}|{float(self.avg_gestalt):.2f}, bin: {float(self.avg_gestalt_mean):.2e}|{np.sqrt(float(self.avg_gestalt2) - float(self.avg_gestalt)**2):.2e} closed: {float(self.avg_update_gestalt):.2e}|{float(self.avg_update_position):.2e}'
        return info
    
class WriterWrapper():

    def __init__(self, use_wandb: bool, cfg: Configuration):
        if use_wandb:
            from torch.utils.tensorboard import SummaryWriter
            import wandb
            wandb.init(project=f'Loci_Looped_{cfg.dataset}', name= cfg.model_path,  sync_tensorboard=True, config=cfg)
            self.writer = SummaryWriter()
        else:
            self.writer = None

    def add_scalar(self, name, value, step):
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)

    def add_video(self, name, value, step):
        if self.writer is not None:
            self.writer.add_video(name, value, step)

    def flush(self):
        if self.writer is not None:
            self.writer.flush()
