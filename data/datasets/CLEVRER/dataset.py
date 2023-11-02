from torch.utils import data
from typing import Tuple, Union, List
import numpy as np
import json
import math
import cv2
import h5py
import os
import pickle


class RamImage():
    def __init__(self, path):
        
        fd = open(path, 'rb')
        img_str = fd.read()
        fd.close()

        self.img_raw = np.frombuffer(img_str, np.uint8)

    def to_numpy(self):
        return cv2.imdecode(self.img_raw, cv2.IMREAD_COLOR) 

class ClevrerSample(data.Dataset):
    def __init__(self, root_path: str, data_path: str, size: Tuple[int, int]):

        self.size = size
        self.data_path = root_path
        self.video_id = int(data_path.split('_')[1])

        frames = []
        for frame in os.listdir(os.path.join(root_path, data_path)):
            if os.path.isfile(os.path.join(root_path, data_path, frame)) and frame.endswith('.jpg'):
                frames.append(os.path.join(root_path, data_path, frame))
        frames.sort()

        self.imgs = []
        for path in frames:
            self.imgs.append(RamImage(path))

    def get_data(self):

        frames = np.zeros((128,3,self.size[1], self.size[0]),dtype=np.float32)
        for i in range(len(self.imgs)):
            img = self.imgs[i].to_numpy()
            frames[i] = img.transpose(2, 0, 1).astype(np.float32) / 255.0

        return frames

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

        return self

class ClevrerDataset(data.Dataset):

    def save(self):
        with open(self.file, "wb") as outfile:
    	    pickle.dump(self.samples, outfile)

    def load(self):
        with open(self.file, "rb") as infile:
            self.samples = pickle.load(infile)

    def __init__(self, root_path: str, dataset_name: str, type: str, size: Tuple[int, int], full_size: Tuple[int, int] = None, use_slotformer: bool = False, evaluation: bool = False):

        data_path  = f'data/data/video/{dataset_name}'
        data_path  = os.path.join(root_path, data_path)
        self.file  = os.path.join(data_path, f'dataset-{size[0]}x{size[1]}-{type}.pickle')

        self.samples = []

        if os.path.exists(self.file):
            self.load()
        else:

            if (full_size is None) or (size == full_size):
                sample_path = os.path.join(data_path, type, 'images')
                samples = []
                print(f'sample_path:', sample_path)

                for directory in os.listdir(sample_path):
                    path = os.path.join(sample_path, directory)
                    if os.path.isdir(path):
                        samples.append(directory)

                samples.sort()
                num_samples = len(samples)
                print(f'num_samples:', num_samples)

                for i, dir in enumerate(samples):
                    self.samples.append(ClevrerSample(sample_path, dir, size))

                    print(f"Loading CLEVRER [{i * 100 / num_samples:.2f}]", flush=True)

            else:
                # load full size dataset
                full_dataset = ClevrerDataset(root_path, dataset_name, type, full_size, full_size)

                # downsample
                for i, sample in enumerate(full_dataset.samples):
                    self.samples.append(sample.downsample(size))

                    print(f"Loading CLEVRER {type} [{i * 100 / len(full_dataset.samples):.2f}]", flush=True)

            self.save()
        
        self.length     = len(self.samples)
        self.background = None
        if "background.jpg" in os.listdir(data_path):
            self.background = cv2.imread(os.path.join(data_path, "background.jpg"))
            self.background = cv2.resize(self.background, dsize=size, interpolation=cv2.INTER_CUBIC)
            self.background = self.background.transpose(2, 0, 1).astype(np.float32) / 255.0
            self.background = self.background.reshape(self.background.shape[0], self.background.shape[1], self.background.shape[2])
        
        self.use_slotformer = use_slotformer
        self.eval = evaluation
        if self.use_slotformer:
            with open(f'{data_path}/slotformer/valid_idx_{type}.pt', 'rb') as f:
                self.slotformer_idx = pickle.load(f)
            self.length = len(self.slotformer_idx)
            self.video_ids = [sample.video_id for sample in self.samples]
            self.burn_in_length = 6
            self.rollout_length = 10
            self.skip_length = 2

            if self.eval:
                self.gt_mask = np.load(f'{data_path}/slotformer/gt_mask.npy').astype(np.int8)             
                self.gt_bbox = np.load(f'{data_path}/slotformer/gt_bbox.npy').astype(np.int16) 
                self.gt_pres_mask = np.load(f'{data_path}/slotformer/gt_pres_mask.npy').astype(bool) 
                #self.gt = np.load(f'{data_path}/slotformer/gt.npy').astype(np.float32)

        print(f"ClevrerDataset: {self.length}")

        if len(self) == 0:
            raise FileNotFoundError(f'Found no dataset at {self.data_path}')
        
        if False:
            for sample in self.samples:
                frames = sample.get_data()
                i = 0
                for frame in frames:
                    i += 1
                    if i == 29:
                        print('test')
                    frame = np.einsum('chw->hwc', frame)
                    cv2.imshow('frame', frame)
                    cv2.waitKey(0) 

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):

        if self.use_slotformer: 
            index_org = index
            _, start_idx, video_id = self.slotformer_idx[index]
            index = self.video_ids.index(video_id)

            #print(f"Loading CLEVRER {index} {start_idx} {video_id}")

            if video_id != self.samples[index].video_id:
                raise ValueError(f"Video ID mismatch {video_id} != {self.samples[index].video_id}")

            selec = range(start_idx, start_idx+(self.burn_in_length+self.rollout_length)*self.skip_length, self.skip_length)
            frames = self.samples[index].get_data()[selec]

            if self.eval:
                gt_mask = self.gt_mask[index_org]
                gt_bbox = self.gt_bbox[index_org]
                gt_pres_mask = self.gt_pres_mask[index_org]
                #gt = self.gt[index_org]

                return (
                    frames,
                    self.background,
                    gt_mask,
                    gt_bbox,
                    gt_pres_mask,
                    #gt
                )

        else:
            frames = self.samples[index].get_data()
            
        return (
            frames,
            self.background
        )

#if __name__ == "__main__":
    #dataset = ClevrerDataset('./', "CLEVRER", "train", [320, 240])

    #dataset = ClevrerDataset('./', "CLEVRER", "train", [480, 320])
    #dataset = ClevrerDataset('./', "CLEVRER", "train", [120, 80], [480, 320])
