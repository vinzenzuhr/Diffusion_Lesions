from DatasetMRI import DatasetMRI
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Tuple
import torch
from scipy.ndimage import label
import os
from tqdm.auto import tqdm

class DatasetMRI2D(DatasetMRI):
    def __init__(self, root_dir_img: Path, root_dir_segm: Path = None, root_dir_masks: Path = None, directDL: bool = True, seed: int = None, only_connected_masks: bool = True):
        super().__init__(root_dir_img, root_dir_segm, root_dir_masks, directDL, seed, only_connected_masks)

        # go through all 3D segmentation and add relevant 2D slices to dict
        idx=0
        for j in tqdm(np.arange(len(self.list_paths_t1n))):
            # if there are masks restrict slices to mask content
            if(self.list_paths_masks):
                for path_mask in self.list_paths_masks[j]:
                    t1n_mask = self._get_mask(path_mask, self.list_paths_segm[j] if self.list_paths_segm else None)
                    if (t1n_mask is None):
                        continue 
 
                    # extract connected components from mask or load them if they are already exist
                    if only_connected_masks:
                        path_component_matrix = os.path.splitext(path_mask)[0] + "_component_matrix.npy"  
                        component_matrix, _ = self._get_component_matrix(t1n_mask, path_component_matrix)

                    # go through every slice and add (connected) masks to dataset  
                    for idx_slice in torch.arange(t1n_mask.shape[1]):
                        if (not t1n_mask[:,idx_slice,:].any()):
                            continue
                        if only_connected_masks:
                            # get connected components in slice and add them to dataset
                            slice_components = list(torch.unique(component_matrix[:, idx_slice, :]))
                            slice_components.remove(torch.tensor(0))

                            relevant_components = []
                            for component in slice_components:
                                min_area = 100
                                if (torch.count_nonzero(component_matrix[:,idx_slice,:]==component) >= min_area):
                                    relevant_components.append(component)
                            if len(relevant_components) > 0:
                                self.idx_to_element[idx]=(self.list_paths_t1n[j], self.list_paths_segm[j] if self.list_paths_segm else None, path_mask, path_component_matrix, relevant_components, idx_slice)
                                idx+=1
                        else:
                            self.idx_to_element[idx]=(self.list_paths_t1n[j], self.list_paths_segm[j] if self.list_paths_segm else None, path_mask, None, None, idx_slice)
                            idx+=1
                    
                    ### calculate n, then go through every component and for every component through every slice
                    #TODO: n speichern und if "only connected masks"
                    """
                    for idx_component in list_component_labels:
                        for idx_slice in torch.arange(t1n_mask.shape[1]):
                            min_area = 100
                            if (torch.count_nonzero(component_matrix[:,idx_slice,:]==idx_component) < min_area):
                                continue 
                            self.idx_to_element[idx]=(self.list_paths_t1n[j], self.list_paths_segm[j] if self.list_paths_segm else None, path_mask, path_component_matrix, idx_component, idx_slice)
                            idx+=1
                    """



            else:
                # if there are no masks, but a segmentation mask restrict slices to white matter regions
                if(self.list_paths_segm):
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
                        None,
                        0,
                        bottom+i)
                    idx+=1
                
    def __getitem__(self, idx): 
            t1n_path = self.idx_to_element[idx][0]
            segm_path = self.idx_to_element[idx][1]
            mask_path = self.idx_to_element[idx][2]
            if self.only_connected_masks:
                component_matrix_path = self.idx_to_element[idx][3]
                components = self.idx_to_element[idx][4]
            slice_idx = self.idx_to_element[idx][5]

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
                t1n_segm = self._padding(t1n_segm.to(torch.uint8))
                t1n_segm_slice = t1n_segm[:,slice_idx,:]
            else:
                t1n_segm_slice = torch.empty(0)   

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

                # pad to pad_shape and get 2D slice from 3D 
                mask = torch.Tensor(mask)
                mask = self._padding(mask.to(torch.uint8)) 
                 
                mask_slice = mask[:,slice_idx,:] 
            else:
                mask_slice = torch.empty(0) 
            # Output data
            sample_dict = {
                "gt_image": t1n_slice.unsqueeze(0),
                "segm": t1n_segm_slice, 
                "mask": mask_slice.unsqueeze(0),
                "max_v": t1n_max_v,
                "idx": int(idx),
                "name": t1n_path.parent.stem,
            } 
            return sample_dict 
