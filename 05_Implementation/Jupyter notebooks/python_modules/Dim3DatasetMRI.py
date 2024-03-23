from DatasetMRI import DatasetMRI
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Tuple
import torch

class Dim3DatasetMRI(DatasetMRI):
    def __init__(self, root_dir_img: Path, root_dir_masks: Path = None, pad_shape=(256,256,256), directDL: bool = False, seed: int = None):
        super().__init__(root_dir_img, root_dir_masks, pad_shape, directDL, seed)

        for i in np.arange(len(self.list_paths_t1n)):
            self.idx_to_element[i]=(self.list_paths_t1n[i], self.list_paths_masks[i] if self.list_paths_masks else None)

    def __getitem__(self, idx): 
            t1n_path = self.idx_to_element[idx][0] 
            mask_path_list = self.idx_to_element[idx][1]

            # load t1n img
            t1n_img = nib.load(t1n_path)
            t1n_img = t1n_img.get_fdata()  

            # preprocess t1n
            t1n_img, t1n_max_v = self.preprocess(t1n_img)  

            # load masks 
            if(mask_path_list): 
                generator = torch.cuda.manual_seed_all(self.seed) if self.seed else None
                rand_idx = torch.randint(high=len(mask_path_list), size=(1,), generator=generator) 
                mask = nib.load(mask_path_list[rand_idx])
                mask = mask.get_fdata() 
                mask = torch.Tensor(mask)
                mask = self.padding(mask)
            else:
                mask = None
            
            # Output data
            sample_dict = {
                "gt_image": t1n_img.unsqueeze(0), 
                "mask": mask.unsqueeze(0),
                "max_v": t1n_max_v,
                "idx": int(idx),
            } 
            return sample_dict