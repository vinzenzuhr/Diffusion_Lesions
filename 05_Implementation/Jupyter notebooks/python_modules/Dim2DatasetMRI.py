from DatasetMRI import DatasetMRI
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Tuple
import torch

class Dim2DatasetMRI(DatasetMRI):
    def __init__(self, root_dir_img: Path, root_dir_segm: Path = None, root_dir_masks: Path = None, pad_shape: Tuple = (256,256,256), directDL: bool = True, seed: int = None):
        super().__init__(root_dir_img, root_dir_segm, root_dir_masks, pad_shape, directDL, seed)
        
        # go through all 3D segmentation and add relevant 2D slices to dict
        idx=0
        for j in np.arange(len(self.list_paths_t1n)):
            # if there are masks restrict slices to mask content
            if(self.list_paths_masks):
                for path_mask in self.list_paths_masks[j]:
                    t1n_mask = nib.load(path_mask)
                    t1n_mask = t1n_mask.get_fdata()  

                    # if there is a segmentation restrict mask to white matter regions
                    if(root_dir_segm):
                        binary_white_matter_segm = self._get_white_matter_segm(self.list_paths_segm[j]) 
                        t1n_mask = binary_white_matter_segm * t1n_mask

                    if (not t1n_mask.any()):
                        print("skip t1n, because no mask inside white matter detected")
                        continue 
 
                    # get first slice with mask content  
                    i=0
                    while(not t1n_mask[:,i,:].any()):
                        i += 1 
                    bottom = i

                    # get last slice with mask content  
                    i=t1n_mask.shape[1]-1
                    while(not t1n_mask[:,i,:].any()):
                        i -= 1 
                    top = i

                    # Add all slices between top and bottom slice to dataset
                    for i in np.arange(top-bottom): 
                        self.idx_to_element[idx]=(
                            self.list_paths_t1n[j], 
                            self.list_paths_segm[j] if self.list_paths_segm else None,
                            path_mask,
                            bottom+i)
                        idx+=1
            else:
                # if there are no masks, but a segmentation mask restrict slices to white matter regions
                if(root_dir_segm):
                    t1n_segm = self._get_white_matter_segm(self.list_paths_segm[j])

                    # get first slice with white matter content  
                    i=0
                    while(not t1n_segm[:,i,:].any()):
                        i += 1 
                    bottom = i
        
                    # get last slice with white matter content  
                    i=t1n_segm.shape[1]-1
                    while(not t1n_segm[:,i,:].any()):
                        i -= 1 
                    top = i
                # if there are no masks and no segmentations don't restrict slices
                else:    
                    t1n_example = nib.load(self.list_paths_t1n[0])
                    t1n_example = t1n_example.get_fdata()
                    bottom = 0
                    top = t1n_example.shape[1]  
                # add all slices between desired top and bottom slice to dataset  
                for i in np.arange(top-bottom): 
                    self.idx_to_element[idx]=(
                        self.list_paths_t1n[j], 
                        self.list_paths_segm[j] if self.list_paths_segm else None, 
                        None, 
                        bottom+i)
                    idx+=1
                
    def __getitem__(self, idx):
            t1n_path = self.idx_to_element[idx][0]
            segm_path = self.idx_to_element[idx][1]
            mask_path = self.idx_to_element[idx][2]
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

                # pad to pad_shape and get 2D slice from 3D
                # make copy to avoid negative strides, which are not supported in Pytorch
                t1n_segm = torch.Tensor(t1n_segm.copy())
                t1n_segm = self._padding(t1n_segm)
                t1n_segm_slice = t1n_segm[:,slice_idx,:]
            else:
                t1n_segm_slice = None  

            # load masks  
            if(mask_path):
                mask = nib.load(mask_path)
                mask = mask.get_fdata()

                # if there is a segmentation restrict mask to white matter regions
                if(segm_path):
                    binary_white_matter_segm = self._get_white_matter_segm(segm_path) 
                    mask = binary_white_matter_segm * mask

                # pad to pad_shape and get 2D slice from 3D
                mask = torch.Tensor(mask)
                mask = self._padding(mask) 
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
