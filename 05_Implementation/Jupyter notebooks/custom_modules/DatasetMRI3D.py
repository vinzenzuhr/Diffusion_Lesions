from DatasetMRI import DatasetMRI
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Tuple
import torch
import os
from scipy.ndimage import label
from tqdm.auto import tqdm

class DatasetMRI3D(DatasetMRI):
    def __init__(self, root_dir_img: Path, root_dir_segm: Path = None, root_dir_masks: Path = None, directDL: bool = True, seed: int = None, only_connected_masks: bool = True):
        super().__init__(root_dir_img, root_dir_segm, root_dir_masks, directDL, seed, only_connected_masks)


        idx=0 
        for i in tqdm(np.arange(len(self.list_paths_t1n))): 
            idx_mask=0
            while True:
                # extract connected components from mask or load them if they are already exist
                if only_connected_masks: 
                    #get mask and calculcate components
                    t1n_mask = self._get_mask(self.list_paths_masks[i][idx_mask], self.list_paths_segm[i])
                    if (t1n_mask is None):
                        continue

                    path_component_matrix = os.path.splitext(self.list_paths_masks[i][idx_mask])[0] + "_component_matrix.npy"
                    component_matrix, n = self._get_component_matrix(t1n_mask, path_component_matrix) 

                    # get only connected masks with a minimum volume
                    min_volume=400#25
                    list_component_labels=[]
                    for j in torch.arange(1, n+1):
                        volume=torch.count_nonzero(component_matrix==j)
                        if volume > min_volume:
                            list_component_labels.append(int(j))
                    
                    # add list of connected masks to dataset
                    if len(list_component_labels) > 0:
                        self.idx_to_element[idx]=(
                            self.list_paths_t1n[i], 
                            self.list_paths_segm[i] if self.list_paths_segm else None, 
                            self.list_paths_masks[i][idx_mask] if self.list_paths_masks else None,
                            path_component_matrix if only_connected_masks else None,
                            list_component_labels if only_connected_masks else None)   
                        idx += 1
                else:
                    self.idx_to_element[idx]=(
                        self.list_paths_t1n[i], 
                        self.list_paths_segm[i] if self.list_paths_segm else None, 
                        self.list_paths_masks[i][idx_mask] if self.list_paths_masks else None,
                        None,
                        None) 
                    idx += 1

                if(self.list_paths_masks and len(self.list_paths_masks[i])-1>idx_mask):
                    idx_mask+=1 
                else:
                    break

    def __getitem__(self, idx): 
            t1n_path = self.idx_to_element[idx][0] 
            segm_path = self.idx_to_element[idx][1]
            mask_path = self.idx_to_element[idx][2]
            if self.only_connected_masks:
                component_matrix_path = self.idx_to_element[idx][3]
                components = self.idx_to_element[idx][4]

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
                if self.only_connected_masks: 
                    # create mask from random connected components
                    component_matrix = torch.load(component_matrix_path) 
                    rand_n = 1 if len(components) == 1 else torch.randint(1, len(components), (1,)).item() 
                    rand_components_idx = torch.randperm(len(components))[:rand_n] 
                    mask = torch.zeros_like(component_matrix)
                    for rand_idx in rand_components_idx: 
                        mask[component_matrix == components[rand_idx]] = 1 
                else:
                    mask = nib.load(mask_path)
                    mask = mask.get_fdata()

                    # if there is a segmentation restrict mask to white matter regions
                    if(segm_path):
                        binary_white_matter_segm = self._get_white_matter_segm(segm_path) 
                        mask = binary_white_matter_segm * mask 

                mask = torch.Tensor(mask)
                mask = self._padding(mask.to(torch.uint8))
                # invert mask, where 0 defines the part to inpaint
                mask = 1-mask 
            else:
                mask = None
            
            # Output data
            sample_dict = {
                "gt_image": t1n_img.unsqueeze(0), 
                "segm": t1n_segm, 
                "mask": mask.unsqueeze(0) if self.list_paths_masks else torch.empty(0), 
                "max_v": t1n_max_v, 
                "idx": int(idx), 
                "name": t1n_path.parent.stem 
            } 
            return sample_dict