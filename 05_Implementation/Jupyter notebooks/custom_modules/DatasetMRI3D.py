import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm.auto import tqdm

from custom_modules import DatasetMRI

class DatasetMRI3D(DatasetMRI):
    """
    A custom dataset class for 3D MRI data.

    Contains ground truth mri images (gt), segmentation tissue maps (segm), lesion masks (masks) 
    and synthesis masks with location where to inpaint lesions.

    Args:
        root_dir_img (Path): The root directory of the MRI images.
        root_dir_segm (Path, optional): The root directory of the segmentation masks. Defaults to None.
        root_dir_masks (Path, optional): The root directory of the masks. Defaults to None.
        root_dir_synthesis (Path, optional): The root directory of the synthesis masks. Defaults to None.
        target_shape (tuple, optional): The target shape to transform the mri img  before slicing. 
            Activating this option is computationally expensive. Defaults to None.
        connected_masks (bool, optional): Whether to use connected masks. Defaults to False.
        min_volume (int, optional): The minimum volume for connected masks. Defaults to 400.
        dilation (int, optional): The dilation value for the masks. Defaults to 0.
        restrict_mask_to_wm (bool, optional): Whether to restrict the mask to white matter regions. 
            Defaults to False.

    Returns:
        __getitem__: A dictionary containing the gt_image, segm, mask, synthesis, idx, name and proc_info. 
    """

    def __init__(
            self, 
            root_dir_img: Path, 
            root_dir_segm: Path = None, 
            root_dir_masks: Path = None, 
            root_dir_synthesis: Path = None, 
            target_shape = None,  
            connected_masks: bool = False,
            min_volume=400,
            dilation=0,
            restrict_mask_to_wm=False):
        super().__init__(
            root_dir_img, 
            root_dir_segm, 
            root_dir_masks, 
            root_dir_synthesis, 
            target_shape,  
            connected_masks, 
            dilation,
            restrict_mask_to_wm)
        self.min_volume=min_volume

        idx_dict = 0 
        for idx_img in tqdm(np.arange(len(self.list_paths_img))): 
            idx_mask = 0
            while True:  
                path_component_matrix = None
                list_component_labels = None
                if connected_masks:
                    path_component_matrix, list_component_labels = self._get_relevant_components(
                        self.list_paths_masks[idx_img][idx_mask], 
                        self.list_paths_img[idx_img], 
                        self.list_paths_segm[idx_img])
                    # if there are no connected components which reach the minimum volume, skip this mask
                    if len(list_component_labels) == 0:
                        idx_mask += 1
                        continue
 
                self.idx_to_element[idx_dict]=(
                    self.list_paths_img[idx_img], 
                    self.list_paths_segm[idx_img] if self.list_paths_segm else None, 
                    self.list_paths_masks[idx_img][idx_mask] if self.list_paths_masks else None,
                    path_component_matrix,
                    list_component_labels,
                    self.list_paths_synthesis[idx_img] if self.list_paths_synthesis else None,) 
                idx_dict += 1

                if(self.list_paths_masks and len(self.list_paths_masks[idx_img]) - 1 > idx_mask):
                    idx_mask += 1 
                else:
                    break
 
    def __getitem__(self, idx):
        """
        Retrieves the item at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the following keys:
                - "gt_image": The preprocessed image as a torch.Tensor.
                - "segm": The preprocessed segmentation as a torch.Tensor.
                - "mask": The preprocessed mask as a torch.Tensor.
                - "synthesis": The preprocessed synthesis mask as a torch.Tensor.
                - "idx": The index of the item.
                - "name": The name of the img folder.
                - "proc_info": A list containing the preprocessing information.
        """
        img_path = self.idx_to_element[idx][0] 
        segm_path = self.idx_to_element[idx][1]
        mask_path = self.idx_to_element[idx][2]
        if self.connected_masks:
            component_matrix_path = self.idx_to_element[idx][3]
            components = self.idx_to_element[idx][4]
        synthesis_path = self.idx_to_element[idx][5] 

        # load objects and preprocess them
        img = nib.load(img_path) 
        mask = nib.load(mask_path) if mask_path else None 
        segm = nib.load(segm_path) if segm_path else None 
        synthesis_mask = nib.load(synthesis_path) if synthesis_path else None 
        img, proc_info, mask, segm, synthesis_mask = self.preprocess(img, mask, segm, synthesis_mask)

        segm = segm if segm_path else torch.empty(0)
        synthesis_mask = synthesis_mask if synthesis_path else torch.empty(0)

        # load masks 
        if(mask_path): 
            if self.connected_masks: 
                # create mask from random connected components
                component_matrix = torch.load(component_matrix_path) 
                rand_n = 1 if len(components) == 1 else torch.randint(1, len(components), (1,)).item() 
                rand_components_idx = torch.randperm(len(components))[:rand_n] 
                mask = torch.zeros_like(component_matrix)
                for rand_idx in rand_components_idx: 
                    mask[component_matrix == components[rand_idx]] = 1
            else: 
                # if there is a segmentation restrict mask to white matter regions
                if(segm_path and self.restrict_mask_to_wm):
                    binary_white_matter_segm = self._get_binary_segm(segm) 
                    mask = binary_white_matter_segm * mask  
        else:
            mask = torch.empty(0) 

        # Output data
        sample_dict = {
            "gt_image": img.unsqueeze(0), 
            "segm": segm, 
            "mask": mask.unsqueeze(0),
            "synthesis": synthesis_mask.unsqueeze(0),
            "idx": int(idx), 
            "name": img_path.parent.stem,
            "proc_info": [proc_info]
        } 
        return sample_dict
    
    def _get_relevant_components(self, path_mask, path_img, path_segm):
        """
        Retrieves the relevant connected components.

        Args:
            path_mask (str): The path to the mask file.
            path_img (str): The path to the image file.
            path_segm (str): The path to the segmentation file.

        Returns:
            tuple: A tuple containing the path to the component matrix file and a list of relevant component labels.

        """
        mask = nib.load(path_mask)         
        img = nib.load(path_img)
        segm = nib.load(path_segm) if path_segm else None
        _, _, mask, segm, _ = self.preprocess(img=img, masks=mask, segm=segm)  

        # if there is a segmentation restrict mask to white matter regions
        if(self.list_paths_segm and self.restrict_mask_to_wm):
            binary_white_matter_segm = self._get_binary_segm(segm)  
            mask = binary_white_matter_segm * mask

        if (not mask.any()):
            return None, [] 

        path_component_matrix = os.path.splitext(path_mask)[0] + "_component_matrix.npy"
        component_matrix, n = self._get_component_matrix(mask, path_component_matrix) 

        # get only connected masks with a minimum volume 
        list_component_labels=[]
        for j in torch.arange(1, n + 1):
            volume=torch.count_nonzero(component_matrix == j)
            if volume > self.min_volume:
                list_component_labels.append(int(j)) 

        return path_component_matrix, list_component_labels