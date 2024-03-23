from pickletools import int4
from torch.utils import data
from typing import Tuple, Union, List
import numpy as np
import json
import math
import cv2
import h5py
import os
import pickle
import sys
import yaml
import warnings
from PIL import Image
from einops import reduce, rearrange, repeat
import torch as th


class BouncingBallDataset(data.Dataset):

    def __init__(self, root_path: str, dataset_name: str, type: str, size: Tuple[int, int], type_name: str = None, full_size: Tuple[int, int] = None, create_dataset: bool = False):

        assert type in ["train", "test", "val"]
        assert type_name in ["interaction", "occlusion", "twolayer", "twolayerdense", "twolayer_ood", "threelayer_ood", "twolayer_ood_3balls"]

        data_path  = f'data/data/video/{dataset_name}'
        data_path  = os.path.join(root_path, data_path)
        self.file  = os.path.join(data_path, f'balls_{type_name}-{type}-{size[0]}x{size[1]}-v1.hdf5')
        self.train = (type == "train")
        self.samples    = []

        if os.path.exists(self.file):
            self.hdf5_file = h5py.File(self.file, "r")
        
        # load dataset
        self.length     = self.hdf5_file['sequence_indices'].shape[0]
        self.background = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        # set number of objects
        if (type_name == "twolayer") or (type_name == "threelayer_ood" and type != "test"):
            self.num_objects = 6
        elif (type_name == "twolayer_ood" and type == "test") or (type_name == "twolayer_ood_3balls" and type == "test"):
            self.num_objects = 4
        elif type_name == "twolayer_ood":
            self.num_objects = 2
        elif type_name == "twolayer_ood_3balls":
            self.num_objects = 3
        elif (type_name == "threelayer_ood" and type == "test"):
            self.num_objects = 9
        else:
            self.num_objects = 3

        if len(self) == 0:
            raise FileNotFoundError(f'Found no dataset at {data_path}')
        
        # loop trough own dataset by calling __getitem__
        if False:
            for i in range(len(self)):
                self[i]

    def add_one_timestep(self, x):
        return np.concatenate((x, np.zeros_like(x[:1])), axis=0)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index: int):

        index_start, length = self.hdf5_file['sequence_indices'][index]
        rgb_images   = self.hdf5_file["rgb_images"][index_start:index_start+length]

        if rgb_images[0].dtype == np.uint8:
            images = []
            for i in range(len(rgb_images)):
                img = cv2.imdecode(rgb_images[i], 1)
                images.append(img.transpose(2, 0, 1).astype(np.float32) / 255.0)

            rgb_images   = np.stack(images)

        rgb_images   = th.from_numpy(rgb_images)

        if self.train:
            return (
                rgb_images,
                self.background
            )

        # EVALUATION
        num_objects = self.num_objects
        instance_positions = self.hdf5_file['instance_positions'][index_start*num_objects:(index_start+length)*num_objects]
        instance_positions = rearrange(instance_positions, '(t o) c -> t o c', o=num_objects)
        instance_positions = instance_positions[:, :, ::-1] # IMPORTANT: flip x and y axis

        instance_pres = self.hdf5_file['instance_incamera'][index_start*num_objects:(index_start+length)*num_objects]
        instance_pres = rearrange(instance_pres, '(t o) c -> t o c', o=num_objects).squeeze(-1)

        instance_bounding_boxes = self.hdf5_file['instance_mask_bboxes'][index_start*num_objects:(index_start+length)*num_objects]
        instance_bounding_boxes = rearrange(instance_bounding_boxes, '(t o) c -> t o c', o=num_objects)
        instance_bounding_boxes = instance_bounding_boxes[:, :, [1, 0, 3, 2]]

        foreground_mask   = self.hdf5_file['foreground_mask'][index_start:(index_start+length)]
        foreground_mask   = rearrange(foreground_mask, 't 1 h w -> t h w')/255

        instance_masks = self.hdf5_file['instance_masks'][index_start*num_objects:(index_start+length)*num_objects]
        instance_masks = rearrange(instance_masks, '(t o) 1 h w -> t o 1 h w', o=num_objects).squeeze()/255

        # CUSTOM
        # use instance masks to to create hidden masks
        hidden_mask = reduce(instance_masks, 't o h w -> t 1 h w', 'sum').squeeze()
        hidden_mask = (hidden_mask > 1).astype(np.uint8)

        # segmentation_masks: index gives which object is visible at that pixel
        segmentation_mask = np.argmax(instance_masks[:, ::-1], axis=1) + 1
        segmentation_mask = foreground_mask * segmentation_mask

        # segmentation mask but only for hidden objects
        segementation_mask_hidden = np.argmax(instance_masks[:, :3], axis=1) + 1 # TODO only works for 6 objects
        segementation_mask_hidden = hidden_mask * segementation_mask_hidden
                
        # add one dummy timestep at the end
        instance_positions = self.add_one_timestep(instance_positions)
        rgb_images = self.add_one_timestep(rgb_images)
        foreground_mask = self.add_one_timestep(foreground_mask)
        hidden_mask = self.add_one_timestep(hidden_mask)
        instance_pres = self.add_one_timestep(instance_pres)
        instance_bounding_boxes = self.add_one_timestep(instance_bounding_boxes)
        instance_masks = self.add_one_timestep(instance_masks)
        segmentation_mask = self.add_one_timestep(segmentation_mask)
        segementation_mask_hidden = self.add_one_timestep(segementation_mask_hidden)

        if False:
            video       = np.array(rgb_images)
            locations   =  np.array(instance_positions)
            fg_masks    = np.array(foreground_mask)
            bb          = np.array(instance_bounding_boxes)
            h_masks     = np.array(hidden_mask)

            # loop through video frames and show them using cv2
            for t in range(video.shape[0]):
                frame = rearrange(video[t], 'c h w -> h w c') * 255

                for loc in locations[t]:
                    x, y = loc
                    #cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1) # did not work properly
                    x_max = int(min(int(x + 2), frame.shape[1]))
                    x_min = int(max(int(x - 2), 0))
                    y_max = int(min(int(y + 2), frame.shape[0]))
                    y_min = int(max(int(y - 2), 0))
                    frame[x_min:x_max, y_min:y_max] = [255, 0, 0]

                # draw the bounding boxes into the frame
                for i, b in enumerate(bb[t]):
                    x_min, y_min, x_max, y_max = b
                    
                    x_min = int(max(x_min, 0))
                    y_min = int(max(y_min, 0))
                    x_max = int(min(x_max, frame.shape[0]-1))
                    y_max = int(min(y_max, frame.shape[0]-1))

                    # dont't use c2 rectangeel function here but draw it manually
                    for pixel in range(x_min, x_max):
                        frame[pixel, y_min, 0] = 255
                        frame[pixel, y_max, 0] = 255
                    for pixel in range(y_min, y_max):
                        frame[x_min, pixel, 0] = 255
                        frame[x_max, pixel, 0] = 255
                
                fg_mask = repeat(fg_masks[t], 'h w -> h w 3') * 255
                h_mask = repeat(h_masks[t], 'h w -> h w 3') * 255
                s_mask = repeat(segmentation_mask[t], 'h w -> h w 3') * (255/6)
                s_mask_hidden = repeat(segementation_mask_hidden[t], 'h w -> h w 3') * (255/3)
                frame = np.concatenate((frame, fg_mask, s_mask, h_mask, s_mask_hidden), axis=1)

                # add instance masks to visualisation
                for i, mask in enumerate(instance_masks[t]):
                    mask = repeat(mask, 'h w -> h w 3') * 255
                    # add border to the right side
                    mask[:, -1, 0] = 255
                    frame = np.concatenate((frame, mask), axis=1)

                frame = frame.astype(np.uint8)
                cv2.imshow('frame', frame)
                cv2.waitKey(0)

        return (
            rgb_images,
            self.background,
            instance_positions,
            segmentation_mask,
            instance_pres,
            segementation_mask_hidden
        )
    

#a = BouncingBallDataset("./", 'BOUNCINGBALLS', "train", (64,64))