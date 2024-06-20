
import os
from pathlib import Path
import random
from typing import Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import label
import torch
from tqdm.auto import tqdm

from custom_modules import DatasetMRI

class DatasetMRI2D(DatasetMRI):
    """
    A custom dataset class for 2D MRI data.

    Contains ground truth mri images (gt), segmentation tissue maps (segm), lesion masks (masks) 
    and synthesis masks with location where to inpaint lesions.

    Args:
        root_dir_img (Path): The root directory of the MRI images.
        restriction (str): The restriction type for selecting slices. Can be 'mask', 'segm', or 'unrestricted'.
        root_dir_segm (Path, optional): The root directory of the segmentation masks. Defaults to None.
        root_dir_masks (Path, optional): The root directory of the masks. Defaults to None.
        root_dir_synthesis (Path, optional): The root directory of the synthesis masks. Defaults to None.
        target_shape (Tuple[int, int, int], optional): The target shape to transform the mri img  before slicing. 
            Activating this option is computationally expensive. Defaults to None.
        connected_masks (bool, optional): Whether to use connected masks. Defaults to False.
        min_area (float, optional): The minimum area for connected masks. Defaults to 100.
        sorted_slice_sample_size (int, optional): The number of sorted slices within one sample. Defaults to 1.
            This is needed for the pseudo3Dmodels, where the model expects that the slices within one batch
            are next to each other in the 3D volume.
        transforms (torch.nn.Module, optional): The transforms to apply to the imgs for augmentation. 
            Transformations only transforms the img and not the mask or segm. Defaults to None.
        dilation (int, optional): The dilation value for the masks. Defaults to 0.
        restrict_mask_to_wm (bool, optional): Whether to restrict the mask to white matter. Defaults to False.
        proportion_training_circular_masks (int, optional): The proportion of training circular masks. Defaults to 0.
        circle_mask_shape (Tuple[int, int], optional): The shape of the circular mask. Defaults to None.
        default_to_circular_mask (bool, optional): Whether to use a circular mask if a mask with no content is provided. 
            Defaults to False.

    Raises:
        ValueError: If root_dir_masks_synthesis is given, then root_dir_masks is mandatory

    Returns:
        __getitem__: A dictionary containing the gt_image, segm, mask, synthesis, idx, idx_slice, name, and mask_name.
            The img and different masks can be one or multiple slices depending on the sorted_slice_sample_size.
    """

    def __init__(
        self, 
        root_dir_img: Path, 
        restriction: str,
        root_dir_segm: Path = None, 
        root_dir_masks: Path = None, 
        root_dir_synthesis: Path = None, 
        target_shape: Tuple[int, int, int] = None,     
        connected_masks: bool = False,
        min_area: float = 100, 
        sorted_slice_sample_size: int = 1,
        transforms: torch.nn.Module = None,
        dilation: int = 0,
        restrict_mask_to_wm: bool = False,
        proportion_training_circular_masks: int = 0,
        circle_mask_shape: Tuple[int, int] = None,
        default_to_circular_mask: bool = False
        ):
        super().__init__(
            root_dir_img, 
            root_dir_segm, 
            root_dir_masks, 
            root_dir_synthesis, 
            target_shape, 
            connected_masks,
            dilation,
            restrict_mask_to_wm)
        
        self.restriction = restriction
        self.sorted_slice_sample_size = sorted_slice_sample_size
        self.transforms = transforms
        self.min_area = min_area
        self.proportionTrainingCircularMasks = proportion_training_circular_masks
        self.circleMaskShape = circle_mask_shape
        self.default_to_circular_mask = default_to_circular_mask
        
        if(root_dir_synthesis and not root_dir_masks):
            raise ValueError(f"If root_dir_masks_synthesis is given, then root_dir_masks is mandatory")

        # go through all 3D segmentation and add relevant 2D slices to dict
        idx_dict=0 
        for idx_img in tqdm(np.arange(len(self.list_paths_img))):  
            slices_dicts = list()
            # if there are a list of masks create a sample per mask
            if(self.list_paths_masks):
                for path_mask in self.list_paths_masks[idx_img]:
                    slices_dicts.append(
                        self._get_relevant_slices(
                            restriction=self.restriction,
                            path_img=self.list_paths_img[idx_img],
                            path_mask=path_mask,
                            path_segm = self.list_paths_segm[idx_img] if self.list_paths_segm else None, 
                            path_synthesis = self.list_paths_synthesis[idx_img] if self.list_paths_synthesis else None,
                        )
                    )
            else:
                slices_dicts.append(
                    self._get_relevant_slices(
                        restriction=self.restriction,
                        path_img = self.list_paths_img[idx_img],
                        path_segm = self.list_paths_segm[idx_img],
                        path_synthesis = self.list_paths_synthesis[idx_img] if self.list_paths_synthesis else None,
                    )
                )  

            for slices_dict in slices_dicts:  
                for i in np.arange(len(slices_dict["list_relevant_slices"])):
                    self.idx_to_element[idx_dict]=(
                        self.list_paths_img[idx_img], 
                        slices_dict["path_segm"], 
                        slices_dict["path_mask"], 
                        slices_dict["path_component_matrix"], 
                        slices_dict["list_relevant_components"][i] if slices_dict["list_relevant_components"] else None,
                        slices_dict["path_synthesis"],
                        slices_dict["list_relevant_slices"][i])
                    idx_dict+=1  
    
    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves the item at the given index from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the following keys:
                - "gt_image": The 2D slice of the image.
                - "segm": The 2D slice of the segmentation mask.
                - "mask": The 2D slice of the lesion or circular mask.
                - "synthesis": The 2D slice of the synthesis mask.
                - "idx": The index of the item.
                - "idx_slice": The index of the slice within the 3D object.
                - "name": The name of the img folder.
                - "mask_name": The name of the mask folder.
        """
        img_path = self.idx_to_element[idx][0]
        segm_path = self.idx_to_element[idx][1]
        mask_path = self.idx_to_element[idx][2]
        if self.connected_masks:
            component_matrix_path = self.idx_to_element[idx][3]
            components = self.idx_to_element[idx][4]
        synthesis_path = self.idx_to_element[idx][5]
        idx_slice = self.idx_to_element[idx][6]

        # load objects and preprocess them
        img = nib.load(img_path)
        mask = nib.load(mask_path) if mask_path else None
        segm = nib.load(segm_path) if segm_path else None
        synthesis_mask = nib.load(synthesis_path) if synthesis_path else None
        img, _, mask, segm, synthesis_mask = self.preprocess(img, mask, segm, synthesis_mask)

        # get 2D slices from 3D objects
        slice = img[:, idx_slice:idx_slice+self.sorted_slice_sample_size, :].permute(1, 0, 2)
        if self.sorted_slice_sample_size > 1:
            slice = slice.unsqueeze(1) 
        if segm != None: 
            segm_slice = segm[:, idx_slice:idx_slice+self.sorted_slice_sample_size, :].permute(1, 0, 2)
            if self.sorted_slice_sample_size > 1:
                segm_slice = segm_slice.unsqueeze(1)
        else:
            segm_slice = torch.empty(0)
        # get 2D circular mask or 2D slice vom 3D lesion mask
        if self.proportionTrainingCircularMasks > random.random():
            mask_slice = self.get_random_circular_masks(n=self.sorted_slice_sample_size) 
        elif mask != None:
            if self.connected_masks:
                # create mask from random amount of connected components
                component_matrix = torch.load(component_matrix_path)
                assert component_matrix.shape == img.shape, f"Component matrix shape {component_matrix.shape} does not match image shape {img.shape}"
                if len(components) == 0:
                    mask_slice = torch.tensor(0) 
                else:
                    rand_n = 1 if len(components) == 1 else torch.randint(1, len(components), (1,)).item()
                    rand_components_idx = torch.randperm(len(components))[:rand_n]
                    mask = torch.zeros_like(component_matrix)
                    for rand_idx in rand_components_idx:
                        mask[component_matrix == components[rand_idx]] = 1
                    mask_slice = mask[:, idx_slice:idx_slice+self.sorted_slice_sample_size, :].permute(1, 0, 2)
            else:
                if segm_path and self.restrict_mask_to_wm:
                    binary_white_matter_segm = self._get_binary_segm(segm)
                    mask = binary_white_matter_segm * mask
                mask_slice = mask[:, idx_slice:idx_slice+self.sorted_slice_sample_size, :].permute(1, 0, 2)

            if not mask_slice.any() and self.default_to_circular_mask:
                mask_slice = self.get_random_circular_masks(n=self.sorted_slice_sample_size)
            
            if self.sorted_slice_sample_size > 1:
                mask_slice = mask_slice.unsqueeze(1)
        else:
            mask_slice = torch.empty(0)
        if synthesis_mask != None:
            synthesis_slice = synthesis_mask[:, idx_slice:idx_slice + self.sorted_slice_sample_size, :].permute(1, 0, 2)
            if self.sorted_slice_sample_size > 1:
                synthesis_slice = synthesis_slice.unsqueeze(1)
        else:
            synthesis_slice = torch.empty(0)

        if self.transforms:
            slice = self.transforms(slice)

        sample_dict = {
            "gt_image": slice,
            "segm": segm_slice,
            "mask": mask_slice,
            "synthesis": synthesis_slice,
            "idx": int(idx),
            "idx_slice": idx_slice,
            "name": img_path.parent.stem,
            "mask_name": mask_path.parent.stem if mask_path else torch.empty(0),
        }
        return sample_dict
    
    def _get_relevant_components(self, component_matrix: torch.tensor, idx_slice: int) -> list:
        """
        Get the relevant components from the component matrix for a given slice index.

        Args:
            component_matrix (torch.tensor): Matrix containing the connected components of the mask.
            idx_slice (int): The index of the slice.

        Returns:
            list: A list of components which are present in the slice and have an area greater than min_area.

        """
        slice_components = list(torch.unique(component_matrix[:, idx_slice, :]))
        slice_components.remove(torch.tensor(0))

        relevant_components = []
        for component in slice_components:
            if (torch.count_nonzero(component_matrix[:, idx_slice, :] == component) >= self.min_area):
                relevant_components.append(component)
        return relevant_components

    def _get_relevant_slices(self, restriction: str, path_img: Path, path_mask: Path = None, 
                             path_segm: Path = None, path_synthesis: Path = None) -> dict:
        """
        Get relevant 2D slices from the 3D input image based on the specified restriction.

        Args:
            restriction (str): The restriction type, either "mask", "segm" or "unrestricted".
                Mask: Only slices with content in the mask are selected e.g. lesions. If connected 
                    masks are used, then only slices with relevant connected components are selected.
                Segm: Only slices with white matter in the segmentation are selected.
                Unrestricted: All slices are selected.
            path_img (Path): The path to the input image.
            path_mask (Path, optional): The path to the mask image. Defaults to None.
            path_segm (Path, optional): The path to the segmentation image. Defaults to None.
            path_synthesis (Path, optional): The path to the synthesis image. Defaults to None.

        Returns:
            dict: A dictionary containing the relevant slices and other information.
        """
        assert (restriction == "mask" and path_img) or (restriction == "segm" and path_segm)  

        output = dict()
        output["path_mask"] = None
        output["path_segm"] = None
        output["path_synthesis"] = None
        output["list_relevant_slices"] = None
        output["path_component_matrix"] = None 
        output["list_relevant_components"] = None
        
        # Load the input images
        img = nib.load(path_img)
        mask = nib.load(path_mask) if path_mask else None
        segm = nib.load(path_segm) if path_segm else None 
        _, _, mask, segm, _ = self.preprocess(img, masks=mask, segm=segm)

        # Get binary white matter segmentations
        if path_segm:
            segm = self._get_binary_segm(segm)    
            if self.restrict_mask_to_wm and path_mask:
                mask = segm * mask

        path_component_matrix = None
        if self.connected_masks:
            path_component_matrix = os.path.splitext(path_mask)[0] + "_component_matrix.npy"  
            component_matrix, _ = self._get_component_matrix(mask, path_component_matrix)

        # Go through every slice and add relevant slices to the dataset. 
        # Skip the last sorted_slice_sample_size slices. They will be added in the __getitem__ function
        list_relevant_slices = list()
        list_relevant_components = list()
        for idx_slice in torch.arange(img.shape[1] - self.sorted_slice_sample_size + 1):
            # If there is no content and it's not in the middle of a package, then skip it
            if restriction == "mask" and not mask[:, idx_slice, :].any():
                continue
            if restriction == "segm" and not segm[:, idx_slice, :].any():
                continue

            relevant_components = None
            if self.connected_masks:
                relevant_components = self._get_relevant_components(component_matrix, idx_slice)
                if restriction == "mask" and len(relevant_components) == 0:
                    continue
            list_relevant_slices.append(idx_slice)
            list_relevant_components.append(relevant_components) 

        output["path_mask"] = path_mask
        output["path_segm"] = path_segm
        output["path_synthesis"] = path_synthesis
        output["list_relevant_slices"] = list_relevant_slices
        output["path_component_matrix"] = path_component_matrix
        output["list_relevant_components"] = list_relevant_components

        return output  
        
    def get_random_circular_masks(self, n: int, generator: torch.Generator = None) -> torch.tensor:
        """
        Generates random circular masks with a random center and radius.

        Args:
            n (int): The number of masks to generate.
            generator (torch.Generator, optional): Generator object for random number generation. Defaults to None.

        Returns:
            torch.tensor: A tensor containing the generated masks.

        """ 
        # chosen by inspection
        lower_bound_radius=3
        upper_bound_radius=50
        std_center=30.0

        center=torch.normal(mean=torch.tensor(self.circleMaskShape).expand(n,2) / 2, std=std_center, generator=generator) 
        radius=torch.rand(n, generator=generator)*(upper_bound_radius-lower_bound_radius) + lower_bound_radius
        
        # create matrix with euclidean distance to center
        Y, X = [torch.arange(self.circleMaskShape[0])[:, None], torch.arange(self.circleMaskShape[1])[None, :]] 
        dist_from_center = torch.sqrt((X.T - center[:, 0])[None, :, :]**2 + (Y - center[:,1])[:, None, :]**2)
        dist_from_center = dist_from_center.permute(2, 0, 1) 
        
        masks = dist_from_center < radius[:, None, None] 
        masks = masks.int() 
        return masks

