from custom_modules import DatasetMRI


import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Tuple
import torch
from scipy.ndimage import label
import os
from tqdm.auto import tqdm
import random

class DatasetMRI2D(DatasetMRI):
    def __init__(
        self, 
        root_dir_img: Path, 
        restriction: str, # 'mask', 'segm', 'unrestricted'
        root_dir_segm: Path = None, 
        root_dir_masks: Path = None, 
        root_dir_synthesis: Path = None, 
        t1n_target_shape = None,     
        only_connected_masks: bool = False,
        min_area = 100, 
        num_sorted_samples=1,
        random_sorted_samples=False,
        transforms=None,
        dilation=0,
        restrict_mask_to_wm=False,
        proportionTrainingCircularMasks=0,
        circleMaskShape = None,
        default_to_circular_mask = False
        ):
        # if num samples >1 the idx_to_element dict is sorted and has packages of num_samples slices which correspond next to each other
        # only transforms which doesn't change the mask or segmentation are allowed

        super().__init__(
            root_dir_img, 
            root_dir_segm, 
            root_dir_masks, 
            root_dir_synthesis, 
            t1n_target_shape, 
            only_connected_masks,
            dilation,
            restrict_mask_to_wm)
        
        self.restriction = restriction
        self.num_sorted_samples = num_sorted_samples
        self.transforms = transforms
        self.min_area = min_area
        self.random_sorted_samples = random_sorted_samples
        self.proportionTrainingCircularMasks = proportionTrainingCircularMasks
        self.circleMaskShape = circleMaskShape
        self.default_to_circular_mask = default_to_circular_mask
        
        if(root_dir_synthesis and not root_dir_masks):
            raise ValueError(f"If root_dir_masks_synthesis is given, then root_dir_masks is mandatory")

        # go through all 3D segmentation and add relevant 2D slices to dict
        idx_dict=0 
        for idx_t1n in tqdm(np.arange(len(self.list_paths_t1n))):  
            slices_dicts = list()

            # if there are a list of masks, add all of them
            if(self.list_paths_masks):
                for path_mask in self.list_paths_masks[idx_t1n]:
                    slices_dicts.append(
                        self._get_relevant_slices(
                            restriction=self.restriction,
                            path_t1n=self.list_paths_t1n[idx_t1n],
                            path_mask=path_mask,
                            path_segm = self.list_paths_segm[idx_t1n] if self.list_paths_segm else None, 
                            path_synthesis = self.list_paths_synthesis[idx_t1n] if self.list_paths_synthesis else None,
                        )
                    )
            else:
                slices_dicts.append(
                    self._get_relevant_slices(
                        restriction=self.restriction,
                        path_t1n = self.list_paths_t1n[idx_t1n],
                        path_segm = self.list_paths_segm[idx_t1n],
                        path_synthesis = self.list_paths_synthesis[idx_t1n] if self.list_paths_synthesis else None,
                    )
                ) 

            for slices_dict in slices_dicts:  
                for i in np.arange(len(slices_dict["list_relevant_slices"]), step=1 if self.random_sorted_samples else num_sorted_samples):

                    # make sure that the next num_samples slices are next to each other
                    if i+num_sorted_samples-1 >= len(slices_dict["list_relevant_slices"]) or (num_sorted_samples>1 and slices_dict["list_relevant_slices"][i+num_sorted_samples-1]-num_sorted_samples+1 != slices_dict["list_relevant_slices"][i]):
                        continue
                    self.idx_to_element[idx_dict]=(
                        self.list_paths_t1n[idx_t1n], 
                        slices_dict["path_segm"], 
                        slices_dict["path_mask"], 
                        slices_dict["path_component_matrix"], 
                        slices_dict["list_relevant_components"][i] if slices_dict["list_relevant_components"] else None, # num samples >1 are not implemented with masks
                        slices_dict["path_synthesis"],
                        slices_dict["list_relevant_slices"][i])
                    idx_dict+=1  
                
    def __getitem__(self, idx): 
            t1n_path = self.idx_to_element[idx][0]
            segm_path = self.idx_to_element[idx][1]
            mask_path = self.idx_to_element[idx][2]
            if self.only_connected_masks:
                component_matrix_path = self.idx_to_element[idx][3]
                components = self.idx_to_element[idx][4]
            synthesis_path = self.idx_to_element[idx][5]
            idx_slice = self.idx_to_element[idx][6] 

            # load t1n img
            t1n_img = nib.load(t1n_path) 
            mask = nib.load(mask_path) if mask_path else None 
            t1n_segm = nib.load(segm_path) if segm_path else None 
            synthesis_mask = nib.load(synthesis_path) if synthesis_path else None

            # preprocess t1n, mask, segm and synthesis
            t1n_img, _, mask, t1n_segm, synthesis_mask = self.preprocess(t1n_img, mask, t1n_segm, synthesis_mask)

            # get 2D slice from 3D  
            t1n_slice = t1n_img[:,idx_slice:idx_slice+self.num_sorted_samples,:].permute(1, 0, 2)
            if self.num_sorted_samples > 1:
                t1n_slice = t1n_slice.unsqueeze(1)
 
            # get 2D slice from 3D segmentation
            if(t1n_segm != None):
                # get 2D slice from 3D 
                t1n_segm_slice = t1n_segm[:,idx_slice:idx_slice+self.num_sorted_samples,:].permute(1, 0, 2)
                if self.num_sorted_samples > 1:
                    t1n_segm_slice = t1n_segm_slice.unsqueeze(1)
            else:
                t1n_segm_slice = torch.empty(0)
             
            if (self.proportionTrainingCircularMasks > random.random()):  
                mask_slice = self.get_random_circular_masks(n=1)
                if self.num_sorted_samples > 1:
                    t1n_segm_slice = t1n_segm_slice.unsqueeze(1)
            elif(mask != None):  
                if self.only_connected_masks: 
                    # create mask from random connected components
                    component_matrix = torch.load(component_matrix_path)
                    assert component_matrix.shape == t1n_img.shape, f"Component matrix shape {component_matrix.shape} does not match t1n image shape {t1n_img.shape}"
                    rand_n = 1 if len(components) == 1 else torch.randint(1, len(components), (1,)).item() 
                    rand_components_idx = torch.randperm(len(components))[:rand_n] 
                    mask = torch.zeros_like(component_matrix)
                    for rand_idx in rand_components_idx: 
                        mask[component_matrix == components[rand_idx]] = 1  
                else: 
                    if(segm_path and self.restrict_mask_to_wm):
                        binary_white_matter_segm = self._get_binary_segm(t1n_segm)
                        mask = binary_white_matter_segm * mask 
                mask_slice = mask[:,idx_slice:idx_slice+self.num_sorted_samples,:].permute(1, 0, 2)

                if not mask_slice.any() and self.default_to_circular_mask:
                    mask_slice = self.get_random_circular_masks(n=1)
                    if self.num_sorted_samples > 1:
                        t1n_segm_slice = t1n_segm_slice.unsqueeze(1)

                if self.num_sorted_samples > 1:
                    mask_slice = mask_slice.unsqueeze(1)
            else:
                mask_slice = torch.empty(0) 
                
            if(synthesis_mask != None): 
                synthesis_slice = synthesis_mask[:,idx_slice:idx_slice+self.num_sorted_samples,:].permute(1, 0, 2)
                if self.num_sorted_samples > 1:
                    synthesis_slice = synthesis_slice.unsqueeze(1)
            else:
                synthesis_slice = torch.empty(0)

            # apply transforms
            if self.transforms:
                t1n_slice = self.transforms(t1n_slice) 

            # Output data
            sample_dict = {
                "gt_image": t1n_slice,
                "segm": t1n_segm_slice, 
                "mask": mask_slice,
                "synthesis": synthesis_slice, 
                "idx": int(idx),
                "idx_slice": idx_slice, 
                "name": t1n_path.parent.stem,
                "mask_name": mask_path.parent.stem if mask_path else torch.empty(0),
            } 
            return sample_dict 
    
    def _get_relevant_components(self, component_matrix, idx_slice): 
        slice_components = list(torch.unique(component_matrix[:, idx_slice, :]))
        slice_components.remove(torch.tensor(0))

        relevant_components = []
        for component in slice_components:
            if (torch.count_nonzero(component_matrix[:,idx_slice,:]==component) >= self.min_area):
                relevant_components.append(component)
        return relevant_components

    def _get_relevant_slices(self, restriction, path_t1n, path_mask = None, path_segm = None, path_synthesis = None):
        # check preconditions 
        assert (restriction == "mask" and path_t1n) or (restriction == "segm" and path_segm)  

        output = dict()
        output["path_mask"] = None
        output["path_segm"] = None
        output["path_synthesis"] = None
        output["list_relevant_slices"] = None
        output["path_component_matrix"] = None 
        output["list_relevant_components"] = None
        
        t1n_img = nib.load(path_t1n)
        t1n_mask = nib.load(path_mask) if path_mask else None
        t1n_segm = nib.load(path_segm) if path_segm else None 
        _, _, t1n_mask, t1n_segm, _ = self.preprocess(t1n_img, masks=t1n_mask, segm=t1n_segm)

        # get binary white matter segmentations
        if path_segm:
            t1n_segm = self._get_binary_segm(t1n_segm)            

            if self.restrict_mask_to_wm and path_mask:
                t1n_mask = t1n_segm * t1n_mask

        # extract connected components from mask or load them if they are already exist
        path_component_matrix = None
        if self.only_connected_masks:
            path_component_matrix = os.path.splitext(path_mask)[0] + "_component_matrix.npy"  
            component_matrix, _ = self._get_component_matrix(t1n_mask, path_component_matrix)

        # go through every slice and add relevant slices to dataset. Make sure that there are at least num_samples slices in a package, which are located next to each other.
        list_relevant_slices = list()
        list_relevant_components = list()
        position_in_package = 0       
        for idx_slice in torch.arange(t1n_img.shape[1]):
            # if there is no relevant content and it's not in the middle of a package then skip it
            if restriction == "mask" and not t1n_mask[:,idx_slice,:].any() and position_in_package == 0:
                continue
            if restriction == "segm" and not t1n_segm[:,idx_slice,:].any() and position_in_package == 0:
                continue

            relevant_components = None
            if self.only_connected_masks:
                relevant_components = self._get_relevant_components(component_matrix, idx_slice)
                if restriction == "mask" and len(relevant_components) == 0 and position_in_package == 0:
                    continue
            list_relevant_slices.append(idx_slice)
            list_relevant_components.append(relevant_components) 

            position_in_package += 1
            if position_in_package == self.num_sorted_samples:
                position_in_package = 0 

        output["path_mask"] = path_mask
        output["path_segm"] = path_segm
        output["path_synthesis"] = path_synthesis
        output["list_relevant_slices"] = list_relevant_slices
        output["path_component_matrix"] = path_component_matrix
        output["list_relevant_components"] = list_relevant_components

        return output  
        
    def get_random_circular_masks(self, n, generator=None):
        #create circular mask with random center around the center point of the pictures and a radius between 3 and 50 pixels
        center=torch.normal(mean=torch.tensor(self.circleMaskShape).expand(n,2)/2, std=30.0, generator=generator) # 30 is chosen by inspection
        low=3
        high=50
        radius=torch.rand(n, generator=generator)*(high-low)+low # get radius between 3 and 50 from uniform distribution 
        
        Y, X = [torch.arange(self.circleMaskShape[0])[:,None],torch.arange(self.circleMaskShape[1])[None,:]] # gives two vectors, each containing the pixel locations. There's a column vector for the column indices and a row vector for the row indices.
        dist_from_center = torch.sqrt((X.T - center[:,0])[None,:,:]**2 + (Y-center[:,1])[:,None,:]**2) # creates matrix with euclidean distance to center
        dist_from_center = dist_from_center.permute(2,0,1) 
        
        masks = dist_from_center < radius[:,None,None] # creates mask for pixels which are inside the radius. 
        masks = masks.int() #masks[:,None,:,:].int() 
        return masks

