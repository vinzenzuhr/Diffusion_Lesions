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
    def __init__(self, root_dir_img: Path, root_dir_segm: Path = None, root_dir_masks: Path = None, root_dir_synthesis: Path = None, directDL: bool = True, seed: int = None, only_connected_masks: bool = True, axis_augmentation: bool = False):
        super().__init__(root_dir_img, root_dir_segm, root_dir_masks, root_dir_synthesis, directDL, seed, only_connected_masks)
        self.axis_augmentation = axis_augmentation
        
        if(root_dir_synthesis and not root_dir_masks):
            raise ValueError(f"If root_dir_masks_synthesis is given, then root_dir_masks is mandatory")

        # go through all 3D segmentation and add relevant 2D slices to dict
        idx_dict=0
        for idx_t1n in tqdm(np.arange(len(self.list_paths_t1n))): 
            # if there are masks restrict slices to mask content
            if(self.list_paths_masks):
                # multiple masks for one t1n image are possible
                for path_mask in self.list_paths_masks[idx_t1n]:
                    idx_dict=self._add_slices_with_mask(
                        path_mask, 
                        self.list_paths_t1n[idx_t1n], 
                        idx_dict, 
                        self.list_paths_segm[idx_t1n] if self.list_paths_segm else None, 
                        self.list_paths_synthesis[idx_t1n] if self.list_paths_synthesis else None)
            # if there are segmentation restrict slices to white matter
            elif(self.list_paths_segm):
                idx_dict = self._add_slices_with_segmentation(self.list_paths_segm[idx_t1n], self.list_paths_t1n[idx_t1n], idx_dict)
            # if there are no masks and no segmentations don't restrict slices  
            else:  
                t1n_example = nib.load(self.list_paths_t1n[idx_t1n])
                t1n_example = t1n_example.get_fdata()
                bottom = 0
                top = t1n_example.shape[1]  
                idx_dict = self._add_slices_between_indices(self.list_paths_t1n[idx_t1n], idx_dict, bottom, top) 
                
                
    def __getitem__(self, idx): 
            t1n_path = self.idx_to_element[idx][0]
            segm_path = self.idx_to_element[idx][1]
            mask_path = self.idx_to_element[idx][2]
            if self.only_connected_masks:
                component_matrix_path = self.idx_to_element[idx][3]
                components = self.idx_to_element[idx][4]
            synthesis_path = self.idx_to_element[idx][5]
            slice_idx = self.idx_to_element[idx][6]

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
            
            if(synthesis_path):
                synthesis_mask = nib.load(synthesis_path)
                synthesis_mask = synthesis_mask.get_fdata()
                synthesis_mask = torch.Tensor(synthesis_mask)
                synthesis_mask = self._padding(synthesis_mask.to(torch.uint8))
                synthesis_slice = synthesis_mask[:,slice_idx,:]
            else:
                synthesis_slice = torch.empty(0)

            # Output data
            sample_dict = {
                "gt_image": t1n_slice.unsqueeze(0),
                "segm": t1n_segm_slice, 
                "mask": mask_slice.unsqueeze(0),
                "synthesis": synthesis_slice.unsqueeze(0),
                "max_v": t1n_max_v,
                "idx": int(idx),
                "slice_idx": int(slice_idx),
                "name": t1n_path.parent.stem,
            } 
            return sample_dict 
    
    def _add_slices_with_mask(self, path_mask, path_t1n, idx_dict, path_segm = None, path_synthesis = None):  

        t1n_mask = self._get_mask(path_mask, path_segm)
        if (t1n_mask is None):
            return idx_dict 

        # extract connected components from mask or load them if they are already exist
        path_component_matrix = None
        if self.only_connected_masks:
            path_component_matrix = os.path.splitext(path_mask)[0] + "_component_matrix.npy"  
            component_matrix, _ = self._get_component_matrix(t1n_mask, path_component_matrix)

        # go through every slice and add (connected) masks to dataset  
        for idx_slice in torch.arange(t1n_mask.shape[1]):
            if (not t1n_mask[:,idx_slice,:].any()):
                continue

            relevant_components = None
            if self.only_connected_masks: 
                relevant_components = self._get_relevant_components(component_matrix, idx_slice)
                if len(relevant_components) == 0:
                    continue
            
            self.idx_to_element[idx_dict]=(
                path_t1n, 
                path_segm, 
                path_mask, 
                path_component_matrix, 
                relevant_components, 
                path_synthesis,
                idx_slice) 
            idx_dict+=1

        return idx_dict

    def _add_slices_with_segmentation(self, path_segm, path_t1n, idx_dict):
        # Restrict slices to white matter regions 
        t1n_segm = self._get_white_matter_segm(path_segm)

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

        # add slices to dict
        idx_dict = self._add_slices_between_indices(path_t1n, idx_dict, bottom, top, path_segm)
        
        return idx_dict
    
    def _add_slices_between_indices(self, path_t1n, idx_dict, bottom, top, path_segm = None):
        # Not usable for masks 
        for i in np.arange(top-bottom): 
            slice_idx = bottom+i
            self.idx_to_element[idx_dict]=(
                path_t1n, 
                path_segm, 
                None, 
                None,
                0,
                0,
                slice_idx)
            idx_dict+=1 
        return idx_dict
    
    def _get_relevant_components(self, component_matrix, idx_slice): 
        slice_components = list(torch.unique(component_matrix[:, idx_slice, :]))
        slice_components.remove(torch.tensor(0))

        relevant_components = []
        for component in slice_components:
            min_area = 100
            if (torch.count_nonzero(component_matrix[:,idx_slice,:]==component) >= min_area):
                relevant_components.append(component)

        return relevant_components

