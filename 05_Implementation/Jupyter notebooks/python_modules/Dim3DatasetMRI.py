from DatasetMRI import DatasetMRI
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Tuple
import torch

class Dim3DatasetMRI(DatasetMRI):
    def __init__(self, root_dir_img: Path, root_dir_segm: Path = None, root_dir_masks: Path = None, pad_shape=(256,256,256), directDL: bool = True, seed: int = None):
        super().__init__(root_dir_img, root_dir_segm, root_dir_masks, pad_shape, directDL, seed)

        idx=0 
        for i in np.arange(len(self.list_paths_t1n)): 
            j=0
            while True:
                self.idx_to_element[idx]=(
                    self.list_paths_t1n[i], 
                    self.list_paths_segm[i] if self.list_paths_segm else None, 
                    self.list_paths_masks[i][j] if self.list_paths_masks else None)
                idx += 1
                
                if(self.list_paths_masks and len(self.list_paths_masks)-1>j):
                    j+=1 
                else:
                    break

    def __getitem__(self, idx): 
            t1n_path = self.idx_to_element[idx][0] 
            segm_path = self.idx_to_element[idx][1]
            mask_path = self.idx_to_element[idx][2]

            # load t1n img
            t1n_img = nib.load(t1n_path)
            t1n_img = t1n_img.get_fdata()  

            # preprocess t1n
            t1n_img, t1n_max_v = self.preprocess(t1n_img)  

            # load segmentation
            if(segm_path):
                t1n_segm = nib.load(segm_path)
                t1n_segm = t1n_segm.get_fdata()

                #transform segmentation if the segmentation came from Direct+DL
                if(self.directDL):
                    t1n_segm = np.transpose(t1n_segm)
                    t1n_segm = np.flip(t1n_segm, axis=1)

                # pad to pad_shape
                # make copy to avoid negative strides, which are not supported in Pytorch
                t1n_segm = torch.Tensor(t1n_segm.copy())
                t1n_segm = self._padding(t1n_segm) 
            else:
                t1n_segm = None  

            # load masks 
            if(mask_path): 
                mask = nib.load(mask_path)
                mask = mask.get_fdata()

                # if there is a segmentation restrict mask to white matter regions
                if(segm_path):
                    binary_white_matter_segm = self._get_white_matter_segm(segm_path) 
                    mask = binary_white_matter_segm * mask 

                # pad to pad_shape
                mask = torch.Tensor(mask)
                mask = self._padding(mask)
            else:
                mask = None
            
            # Output data
            sample_dict = {
                "gt_image": t1n_img.unsqueeze(0), 
                "segm": t1n_segm, 
                "mask": mask.unsqueeze(0), 
                "max_v": t1n_max_v,
                "idx": int(idx),
            } 
            return sample_dict