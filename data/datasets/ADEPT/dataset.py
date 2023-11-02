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
from einops import reduce, rearrange
from scripts.utils.plot_utils import get_color

class RamImage():
    def __init__(self, path):
        
        fd = open(path, 'rb')
        img_str = fd.read()
        fd.close()

        self.img_raw = np.frombuffer(img_str, np.uint8)

    def to_numpy(self):
        return cv2.imdecode(self.img_raw, cv2.IMREAD_COLOR) 

class AdeptSample(data.Dataset):
    def __init__(self, root_path: str, data_path: str, size: Tuple[int, int], type: str):

        data_path = os.path.join(root_path, data_path)
        image_path = os.path.join(data_path, 'imgs')
        self.data_path = data_path
        self.size = size
        self.imgs = []
        self.num_objects = 0

        frames = []
        for file in os.listdir(image_path):
            if (file.startswith("train") or file.startswith("human")) and (file.endswith(".jpg")):
                frames.append(os.path.join(image_path, file))

        frames.sort()
        for i,path in enumerate(frames):
            self.imgs.append(RamImage(path))

        # load config 
        config = self.load_config()
        self.num_objects = len(config['scene'][0]['objects'])

        # add background image
        bg_masks, object_visibility, self.intact = self.compute_background_masks(data_path)        
        self.compute_mean_background(bg_masks)

        # extract object information
        self.extract_objects(config, object_visibility)
        self.compute_unique_positions()

        # extract suprise 
        suprise_dict = {'block': [1], 'delay': [1], 'disappear': [1,2], 'disappear_fixed': [1,2], 'discontinuous': [1,2], 'overturn': [0,3]}
        if type in ['train','test']:
            self.is_suprising = False
        elif type in suprise_dict:
            case_name = config['case_name']
            self.case_name = case_name
            self.case = int(case_name[-1])
            self.is_suprising = (self.case in suprise_dict[type])
        else:
            raise Exception(f'Unknown surprise type: {type}')


    def extract_objects(self, config, object_visibility):

        self.check_camera_config(config['camera'])

        # per object
        self.objects = []
        self.object_types = []
        self.object_colors = []

        # per scene
        self.object_positions = []
        self.object_visibility = []

        for f,frame in enumerate(config['scene']):
            positions = []
            visibility = []
            for o,object in enumerate(frame['objects']):
                if f == 0:
                    self.objects.append(object['name'])
                    self.object_types.append(object['type'])
                    self.object_colors.append(object['color'])

                positions.append(self.get_camera_coords(object['location']))
                if f < len(object_visibility) and o < len(object_visibility[f]):
                    visibility.append(object_visibility[f][o])
                else:
                    visibility.append(False)
            self.object_positions.append(positions)
            self.object_visibility.append(visibility)

        pass

    def load_config(self):

        # load yaml config file
        config = None
        for file in os.listdir(self.data_path):
            if file.endswith(".yaml"):
                with open(os.path.join(self.data_path, file)) as f:
                    try:
                        config = yaml.safe_load(f)   
                    except yaml.YAMLError as exc:
                        print(exc)

        # extract data from yaml file
        if config is None:
            raise Exception(f'No config file found: {self.data_path}')

        return config

    def downsample(self, size):
        self.size = size
        imgs = []
        path = os.path.join(self.data_path, 'tmp.jpg')
        for image_large in self.imgs:
            img_small = cv2.resize(image_large.to_numpy(), dsize=(self.size[0], self.size[1]), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(path, img_small)
            imgs.append(RamImage(path))
        self.imgs = imgs

        # remove tmp.jpg
        os.remove(path)

        # downsample background image
        background = rearrange(self.background, 'c h w -> h w c')
        background = cv2.resize(background, dsize=(self.size[0], self.size[1]), interpolation=cv2.INTER_CUBIC)
        self.background = rearrange(background, 'h w c -> c h w')

        return self

    def get_data(self):

        frames = np.zeros((len(self.imgs),3,self.size[1], self.size[0]),dtype=np.float32)
        for i in range(len(self.imgs)):
            img = self.imgs[i].to_numpy()
            frames[i] = img.transpose(2, 0, 1).astype(np.float32) / 255.0

        return frames

    def compute_color_background_masks(self):

        # access frames
        frames = self.get_data()

        # add 
        background_masks = []
        
        for frame in frames:

            # filter for color 
            mask = np.abs(np.min(frame, axis=0) - np.max(frame, axis=0)) > 0.08

            # display for control
            if False:
                # convert flow_mask to rgb
                flow_mask_rgb = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
                flow_mask_rgb[mask] = [255, 255, 255]
                Horiz = np.concatenate((np.einsum('chw->hwc', frame), flow_mask_rgb), axis=1)
                cv2.imshow('frame', Horiz)
                cv2.waitKey(0) 

            # convert and store 
            background_mask =  (1 - mask).astype(np.byte)
            background_masks.append(background_mask)

        return background_masks

    def compute_mean_background(self, bg_mask):

        # access frames
        frames = self.get_data()
        bg_mask = np.expand_dims(bg_mask, axis=1)

        # mean background masked with bg mask 
        mean_background = np.mean(frames, axis=0, where=bg_mask == 1)

        # replace nan with 0.6
        mean_background = np.nan_to_num(mean_background, nan=0.6)

        # replace black spots with 0.6
        mask = np.mean(mean_background, axis=0) < 0.2
        mean_background[:, mask] = 0.6

        # display for control
        if False:
            for frame,mask in zip(frames, bg_mask):
                mask = np.repeat(mask, 3, axis=0)
                Horiz = np.concatenate((np.einsum('chw->hwc', frame), np.einsum('chw->hwc', mask)), axis=1)
                cv2.imshow('frame', Horiz)
                cv2.waitKey(0) 
  
        self.background = mean_background

        pass

    def compute_background_masks(self, data_path):

        intact = True
        frames = []
        mask_path = os.path.join(data_path, 'masks')        
        for file in os.listdir(mask_path):

            if (file.endswith(".jpg") or file.endswith(".png")):
                ending = file.split('_')[-1]

                # bug in data: handle errorneous mask files 
                if len(str(ending)) == 11:
                    object_id = int(file.split('_')[-2])
                    if object_id < self.num_objects:
                        frames.append(os.path.join(mask_path, file))

        frames.sort()

        # bug in data: handle the additional/errorneous mask file that is added to some of the mask directories
        mod = len(frames) % len(self.imgs)
        if mod > 0:
            frames = frames[:-mod]
            print(f'Warning: Removed {mod} frames from {data_path} due to errorneous mask files.')
            intact = False

        bg_masks = []
        for i,path in enumerate(frames):
            bg_mask = 1 - np.array(Image.open(path)).max(axis=2) / 255.0
            bg_masks.append(bg_mask)

        # add masks of all objects for each frame
        bg_masks = rearrange(np.array(bg_masks), '(o l) h w -> l o h w', l = len(self.imgs))

        # compute if object is present in frame
        object_visibility = reduce(bg_masks, 'l o h w -> l o', 'min') == 0

        # add masks of all objects for each frame
        bg_masks = reduce(bg_masks, 'l o h w -> l h w', 'min')

        return bg_masks, object_visibility, intact

    def check_camera_config(self, camera):
        if camera['camera_look_at'] != [-1.5, 0, 0]:
            print(camera)
            raise Exception(f'Camera look_at is not [-1.5, 0, 0] for sample {self.data_path}')

        if camera['camera_phi'] != 0:
            print(camera)
            raise Exception(f'Camera phi is not 0 for sample {self.data_path}')
        
        if camera['camera_rho'] != 7.2:
            print(camera)
            raise Exception(f'Camera rho is not 7.2 for sample {self.data_path}')

        if camera['camera_theta'] != 20:
            print(camera)
            raise Exception(f'Camera theta is not 20 for sample {self.data_path}')

    # converts the blender coordinates to our camera coordinates
    def get_camera_coords(self, coord):

        camera_matrix = np.array([[ 0.0000,  1.0000,  0.0000, -0.0000], [-0.3420,  0.0000,  0.9397, -0.5130], [ 0.9397, -0.0000,  0.3420, -5.7905], [-0.0000,  0.0000, -0.0000,  1.0000]])
        frame = [[0.5, 0.3611111044883728, -1.09375], [0.5, -0.3611111044883728, -1.09375], [-0.5, -0.3611111044883728, -1.09375]]

        coord = coord + [1]
        co_local = camera_matrix @ coord
        co_local = co_local[:3]
        z = -co_local[2]

        if z == 0.0:
            camera_coords =  [0.5, 0.5, 0.0]
        else:
            frame = [-(v / (v[2] / z)) for v in frame]

            min_x, max_x = frame[2][0], frame[1][0]
            min_y, max_y = frame[1][1], frame[0][1]

            x = (co_local[0]- min_x) / (max_x - min_x)
            y = (co_local[1] - min_y) / (max_y - min_y)

            camera_coords = [x,y,z]

        # determine visibility
        #visible = camera_coords[2] > 0.0 and camera_coords[0] > 0.0 and camera_coords[0] < 1.0 and camera_coords[1] > 0.0 and camera_coords[1] < 1.0

        # revert y axis
        camera_coords[1] = 1-camera_coords[1]

        # switch x and y axis
        camera_coords = [camera_coords[1], camera_coords[0], camera_coords[2]]

        # convert to -1 to 1 scale
        camera_coords = (np.array(camera_coords) - 0.5) * 2.0

        return camera_coords

    def compute_unique_positions(self):

        # per scene
        self.odd_motion = False

        # per object
        new_object_types = []
        new_object_colors = []

        # per frame
        new_object_positions = []
        new_object_visibility = []

        # get rid of duplicate object names
        unique_names = []
        mapping = []
        for i,object in enumerate(self.objects):

            # get identifier of object
            name = object.split('_')[0]
            color = self.object_colors[i]
            name = name + '_' + color

            if name not in unique_names:
                unique_names.append(name)
                new_object_types.append(self.object_types[i])
                new_object_colors.append(self.object_colors[i])
                mapping.append([i])
            else:
                self.odd_motion = True
                mapping[unique_names.index(name)].append(i)

        # loop though all frames and update positions
        for object_positions_of_frame, visibility_of_frame in zip(self.object_positions, self.object_visibility):
            positions = []
            visibility = []

            for c,candidates_per_object in enumerate(mapping):
                
                # unique object and defualt
                candidate_winning = candidates_per_object[0]

                # duplicated object
                if len(candidates_per_object) > 1:
                    for candidate in candidates_per_object:
                        candidate_position = object_positions_of_frame[candidate]
                        candidate_position = candidate_position/2 + 0.5
                        if candidate_position[2] > 0.0 and candidate_position[0] > 0.0 and candidate_position[0] < 1.0 and candidate_position[1] > 0.0 and candidate_position[1] < 1.0:
                            candidate_winning = candidate
                            break    

                # add to new list
                positions.append(object_positions_of_frame[candidate_winning])
                visibility.append(visibility_of_frame[candidate_winning])

            new_object_positions.append(positions)
            new_object_visibility.append(visibility)

        # update self
        self.object_positions = new_object_positions
        self.object_visibility = new_object_visibility
        self.objects = unique_names
        self.object_types = new_object_types
        self.object_colors = new_object_colors

        pass


class AdeptDataset(data.Dataset):

    def save(self):
        state = { 'samples': self.samples }
        with open(self.file, "wb") as outfile:
    	    pickle.dump(state, outfile)

    def load(self):
        with open(self.file, "rb") as infile:
            state = pickle.load(infile)
            self.samples = state['samples']

    def __init__(self, root_path: str, dataset_name: str, type: str, size: Tuple[int, int], type_name: str = None, full_size: Tuple[int, int] = None, create_dataset: bool = False):

        if type_name is None:
            type_name = type

        data_path  = f'data/data/video/{dataset_name}'
        data_path  = os.path.join(root_path, data_path)
        self.file  = os.path.join(data_path, f'dataset-{size[0]}x{size[1]}-{type_name}.pickle')
        self.train = (type == "train")
        self.samples    = []

        if os.path.exists(self.file) and not create_dataset:
            self.load()
        else:

            warnings.filterwarnings(action='ignore', message='Mean of empty slice')
            warnings.filterwarnings(action='ignore', message='invalid value encountered in divide')

            if (full_size is None) or (size == full_size):
                if type in ['train', 'test', 'val']:
                    data_path = os.path.join(data_path, 'training')
                    samples         = list(filter(lambda x: x.startswith("train"), next(os.walk(data_path))[1]))
                else:
                    # distinguish different scenarios and special case for dissapear and diasppear_fixed
                    data_path = os.path.join(data_path, 'human')
                    samples = list(filter(lambda x: type in x and ((type != "disappear") or not ('fixed' in x)), next(os.walk(data_path))[1]))
                num_all_samples = len(samples)

                if type == "train":
                    num_samples = int(num_all_samples * 0.9)
                    sample_start = 0
                elif type == "test" or type == "val":
                    num_samples = int(num_all_samples * 0.1)
                    sample_start = int(num_all_samples * 0.9)
                else: 
                    num_samples  = num_all_samples 
                    sample_start = 0

                for i, dir in enumerate(samples[sample_start:sample_start+num_samples]):
                    self.samples.append(AdeptSample(data_path, dir, size, type))

                    print(f"Loading ADEPT {type} [{i * 100 / num_samples:.2f}]", flush=True)

            else:
                # load full size dataset
                full_dataset = AdeptDataset(root_path, dataset_name, type, full_size, type_name, full_size)

                # downsample
                for i, sample in enumerate(full_dataset.samples):
                    self.samples.append(sample.downsample(size))

                    print(f"Loading ADEPT {type} [{i * 100 / len(full_dataset.samples):.2f}]", flush=True)

            self.save()
        
        self.length     = len(self.samples)
        self.background = None

        if False:
            for sample in self.samples:
                frame = sample.get_data()[0]
                frame = np.concatenate((np.einsum('chw->hwc', frame), np.einsum('chw->hwc', sample.background)), axis=1)
                cv2.imshow('frame', frame)
                cv2.waitKey(0) 
                
        if False:
            counter = 0
            for i, sample in enumerate(self.samples):
                frames = sample.get_data()
                j = 0

                print()
                print('Sample', counter)
                counter += 1

                while j < len(frames):

                    # overwrite last print
                    sys.stdout.flush()
                    sys.stdout.write("\r" + 'Frame: ' + str(j))

                    frame = frames[j]
                    frame = np.einsum('chw->hwc', frame)

                    # add object positions
                    object_positions_frame = sample.object_positions[j]
                    object_visibility_frame = sample.object_visibility[j]
                    for pos_index, position in enumerate(object_positions_frame):
                        position = position/2 + 0.5

                        #if not object_visibility_frame[pos_index]:
                        if position[2] > 0.0 and position[0] > 0.0 and position[0] < 1.0 and position[1] > 0.0 and position[1] < 1.0:
                            h = int(position[0]*frame.shape[0])
                            w = int(position[1]*frame.shape[1])
                            if h > 5 and h < frame.shape[0]-5 and w > 5 and w < frame.shape[1]-5:
                                frame[(h-5):(h+5), (w-5):(w+5), :] = get_color(pos_index)

                    cv2.imshow('frame', frame)

                    if True:
                        # wait for a second 
                        cv2.waitKey(30)
                        j += 1
                        
                    else:
                        # if right error is pressed move on frame in the future, if left error is pressed move on frame in the past
                        keys = cv2.waitKey(10) & 0xFF
                        if keys == ord('q'):
                            j -= 1
                        elif keys == ord('w'):
                            j += 1

        print(f"AdeptDataset[{type}]: {self.length}")

        if len(self) == 0:
            raise FileNotFoundError(f'Found no dataset at {data_path}')

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):

        if self.train:
            return (
                self.samples[index].get_data(),
                self.samples[index].background
            )
        
        occluder_mask = np.array([el == 'Occluder' for el in self.samples[index].object_types])

        return (
            self.samples[index].get_data(),
            self.samples[index].background,
            self.samples[index].is_suprising,
            np.array(self.samples[index].object_positions),
            np.array(self.samples[index].object_visibility),
            occluder_mask
        )