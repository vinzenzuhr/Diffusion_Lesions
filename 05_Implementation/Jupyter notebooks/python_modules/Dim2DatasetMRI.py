from DatasetMRI import DatasetMRI
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Tuple
import torch

class Dim2DatasetMRI(DatasetMRI):
    def __init__(self, root_dir_img: Path, root_dir_segm: Path = None, root_dir_masks: Path = None, pad_shape: Tuple = (256,256,256), directDL: bool = False, seed: int = None):
        super().__init__(root_dir_img, root_dir_masks, pad_shape, directDL, seed)
        self.list_paths_segm = list(root_dir_segm.rglob("*.nii.gz"))

        if(root_dir_segm and (len(self.list_paths_t1n) != len(self.list_paths_segm))):
            raise ValueError(f"The amount of T1n files and segm files must be the same. Got {len(self.list_paths_t1n)} and {len(self.list_paths_segm)}")        
 
        #define offsets between first and last segmented slices and the slices to be used for training
        bottom_offset=60
        top_offset=20
        #go through all 3D segmentation and add relevant 2D slices to dict
        idx=0
        for j in np.arange(len(self.list_paths_t1n)):
            if(root_dir_segm):
                t1n_segm = nib.load(self.list_paths_segm[j])
                t1n_segm = t1n_segm.get_fdata()
    
                #transform segmentation if the segmentation came from Direct+DL
                if self.directDL:
                    t1n_segm = np.transpose(t1n_segm)
                    t1n_segm = np.flip(t1n_segm, axis=1)
    
                #get first slice with segmented content plus offset
                i=0
                while(not t1n_segm[:,i,:].any()):
                    i += 1
                bottom = i + bottom_offset 
    
                #get last slice with segmented content minus offset
                i=t1n_segm.shape[1]-1
                while(not t1n_segm[:,i,:].any()):
                    i -= 1
                top = i - top_offset 
            else:
                t1n_example = nib.load(self.list_paths_t1n[0])
                t1n_example = t1n_example.get_fdata()
                bottom = 0
                top = t1n_example.shape[1]
            #Add all slices between desired top and bottom slice to dataset  
            for i in np.arange(top-bottom): 
                self.idx_to_element[idx]=(
                    self.list_paths_t1n[j], 
                    self.list_paths_segm[j] if self.list_paths_segm else None, 
                    self.list_paths_masks[j] if self.list_paths_masks else None, 
                    bottom+i)
                idx+=1
                
    def __getitem__(self, idx):
            t1n_path = self.idx_to_element[idx][0]
            segm_path = self.idx_to_element[idx][1]
            mask_path_list = self.idx_to_element[idx][2]
            slice_idx = self.idx_to_element[idx][3]

            # load t1n img
            t1n_img = nib.load(t1n_path)
            t1n_img = t1n_img.get_fdata()

            # preprocess t1n
            t1n_img, t1n_max_v = self.preprocess(t1n_img)

            # get 2D slice from 3D
            t1n_slice = t1n_img[:,slice_idx,:]  

            # load segmentation
            if(segm_path):
                t1n_segm = nib.load(segm_path)
                t1n_segm = t1n_segm.get_fdata()

                #transform segmentation if the segmentation came from Direct+DL
                if(self.directDL):
                    t1n_segm = np.transpose(t1n_segm)
                    t1n_segm = np.flip(t1n_segm, axis=1)

                # get 2D slice from 3D
                t1n_segm_slice = t1n_segm[:,slice_idx,:]
            else:
                t1n_segm_slice = None  

            # load masks  
            if(mask_path_list): 
                generator = torch.cuda.manual_seed_all(self.seed) if self.seed else None
                rand_idx = torch.randint(high=len(mask_path_list), size=(1,), generator=generator) 
                mask = nib.load(mask_path_list[rand_idx])
                mask = mask.get_fdata()
                mask = torch.Tensor(mask)
                mask = self.padding(mask)
                mask_slice = mask[:,slice_idx,:] 
            else:
                mask_slice = None
 
            
            # Output data
            sample_dict = {
                "gt_image": t1n_slice.unsqueeze(0),
                "segm": t1n_segm_slice, 
                "mask": mask_slice.unsqueeze(0),
                "max_v": t1n_max_v,
                "idx": int(idx),
            }
            return sample_dict 