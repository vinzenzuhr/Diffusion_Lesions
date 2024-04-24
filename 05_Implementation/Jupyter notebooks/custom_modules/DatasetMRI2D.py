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
    def __init__(
        self, 
        root_dir_img: Path, 
        root_dir_segm: Path = None, 
        root_dir_masks: Path = None, 
        root_dir_synthesis: Path = None, 
        directDL: bool = True, 
        only_connected_masks: bool = True, 
        num_samples=1,
        transforms=None):
        # if num samples >1 the idx_to_element dict is sorted and has packages of num_samples slices which correspond next to each other
        # only transforms which doesn't change the mask or segmentation are allowed

        super().__init__(
            root_dir_img, 
            root_dir_segm, 
            root_dir_masks, 
            root_dir_synthesis, 
            directDL, 
            only_connected_masks)
        
        self.num_samples = num_samples
        self.transforms = transforms
        
        if(root_dir_synthesis and not root_dir_masks):
            raise ValueError(f"If root_dir_masks_synthesis is given, then root_dir_masks is mandatory")

        # go through all 3D segmentation and add relevant 2D slices to dict
        idx_dict=0 
        for idx_t1n in tqdm(np.arange(len(self.list_paths_t1n))): 
            slices_dicts = list()            
            # if there are masks restrict slices to mask content
            if(self.list_paths_masks): 
                # multiple masks for one t1n image are possible
                for path_mask in self.list_paths_masks[idx_t1n]:
                    slices_dicts.append(
                        self._get_relevant_slices(
                            "mask", 
                            path_mask = path_mask, 
                            path_segm = self.list_paths_segm[idx_t1n] if self.list_paths_segm else None, 
                            path_synthesis = self.list_paths_synthesis[idx_t1n] if self.list_paths_synthesis else None)) 
            # if there are segmentation restrict slices to white matter
            elif(self.list_paths_segm):
                slices_dicts.append(
                    self._get_relevant_slices(
                        procedure = "segmentation",  
                        path_segm = self.list_paths_segm[idx_t1n]))  
            # if there are no masks and no segmentations don't restrict slices  
            else:  
                t1n_example = nib.load(self.list_paths_t1n[idx_t1n])
                t1n_example = t1n_example.get_fdata()
                slices_dicts.append({
                    "path_mask": None,
                    "path_segm": None,
                    "path_synthesis": None,
                    "list_relevant_slices": np.arange(0, t1n_example.shape[1]),
                    "path_component_matrix": None,
                    "list_relevant_components": None
                }) 

            for slices_dict in slices_dicts:
                for i in np.arange(len(slices_dict["list_relevant_slices"]), step=num_samples):
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
            #t1n_img = t1n_img.get_fdata()

            # preprocess t1n
            t1n_img, _ = self.preprocess(t1n_img)

            # get 2D slice from 3D 
            #t1n_slice = t1n_img.index_select(dim=1, index=torch.tensor(slice_indices))
            t1n_slice = t1n_img[:,idx_slice:idx_slice+self.num_samples,:].permute(1, 0, 2)
            if self.num_samples > 1:
                t1n_slice = t1n_slice.unsqueeze(1)

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
                t1n_segm = self.resize_image(t1n_segm)
                t1n_segm = self._padding(t1n_segm)
                t1n_segm_slice = t1n_segm[:,idx_slice:idx_slice+self.num_samples,:].reshape(-1, self.t1n_target_shape[0], self.t1n_target_shape[2])
                if self.num_samples > 1:
                    t1n_segm_slice = t1n_segm_slice.unsqueeze(1)
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
                mask = self.resize_image(mask)
                mask = self._padding(mask.to(torch.uint8)) 
                 
                mask_slice = mask[:,idx_slice:idx_slice+self.num_samples,:].reshape(-1, self.t1n_target_shape[0], self.t1n_target_shape[2])
                if self.num_samples > 1:
                    mask_slice = mask_slice.unsqueeze(1)
            else:
                mask_slice = torch.empty(0)   
            
            if(synthesis_path):
                synthesis_mask = nib.load(synthesis_path)
                synthesis_mask = synthesis_mask.get_fdata()
                synthesis_mask = torch.Tensor(synthesis_mask)
                synthesis_mask = self.resize_image(synthesis_mask)
                synthesis_mask = self._padding(synthesis_mask.to(torch.uint8))
                synthesis_slice = synthesis_mask[:,idx_slice:idx_slice+self.num_samples,:]
                if self.num_samples > 1:
                    synthesis_slice = synthesis_slice.reshape(-1, 1, self.t1n_target_shape[0], self.t1n_target_shape[2])
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
            } 
            return sample_dict 
    
    def _get_relevant_components(self, component_matrix, idx_slice): 
        slice_components = list(torch.unique(component_matrix[:, idx_slice, :]))
        slice_components.remove(torch.tensor(0))

        relevant_components = []
        for component in slice_components:
            min_area = 100
            if (torch.count_nonzero(component_matrix[:,idx_slice,:]==component) >= min_area):
                relevant_components.append(component)

        return relevant_components

    def _get_relevant_slices(self, procedure, path_mask = None, path_segm = None, path_synthesis = None):
        output = dict()
        output["path_mask"] = None
        output["path_segm"] = None
        output["path_synthesis"] = None
        output["list_relevant_slices"] = None
        output["path_component_matrix"] = None 
        output["list_relevant_components"] = None

        if procedure == "mask":
            assert path_mask, "If procedure is mask, then path_mask is mandatory" 

            t1n_mask = self._get_mask(path_mask, path_segm)
            if (t1n_mask is None):
                return None, [], [] 

            # extract connected components from mask or load them if they are already exist
            path_component_matrix = None
            if self.only_connected_masks:
                path_component_matrix = os.path.splitext(path_mask)[0] + "_component_matrix.npy"  
                component_matrix, _ = self._get_component_matrix(t1n_mask, path_component_matrix)

            # go through every slice and add (connected) masks to dataset. Make sure that there are at least num_samples slices in a package, which are located next to each other.
            list_relevant_slices = list()
            list_relevant_components = list()
            position_in_package = 0
            for idx_slice in torch.arange(t1n_mask.shape[1]):
                if not t1n_mask[:,idx_slice,:].any() and position_in_package == 0:
                    continue

                relevant_components = None
                if self.only_connected_masks: 
                    relevant_components = self._get_relevant_components(component_matrix, idx_slice)
                    if len(relevant_components) == 0 and position_in_package == 0:
                        continue
                list_relevant_slices.append(idx_slice)
                list_relevant_components.append(relevant_components)

                position_in_package += 1
                if position_in_package == self.num_samples:
                    position_in_package = 0 

            output["path_mask"] = path_mask
            output["path_segm"] = path_segm
            output["path_synthesis"] = path_synthesis
            output["list_relevant_slices"] = list_relevant_slices
            output["path_component_matrix"] = path_component_matrix
            output["list_relevant_components"] = list_relevant_components

            return output 
        
        elif procedure == "segmentation":
            assert path_segm, "If procedure is segmentation, then path_segm is mandatory"

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
            top = i+1  

            output["path_segm"] = path_segm
            output["list_relevant_slices"] = np.arange(bottom, top)
            
            return output
        else:
            raise ValueError(f"Procedure {procedure} is not supported")

