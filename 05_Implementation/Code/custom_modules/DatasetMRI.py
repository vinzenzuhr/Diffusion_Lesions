from abc import abstractmethod 
from pathlib import Path
from typing import Tuple
import os

import nibabel as nib
import nibabel.processing
import numpy as np
from scipy.ndimage import label
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

class DatasetMRI(Dataset):
    """
    Dataset for Training purposes. 
    
    Contains ground truth mri images (gt), segmentation tissue maps (segm), lesion masks (masks) 
    and synthesis masks with location where to inpaint lesions.

    Args:
        root_dir_img (Path): Path to img files
        root_dir_segm (Path, optional): Path to segmentation maps. Defaults to None.
        root_dir_masks (Path, optional): Path to mask files. 
            There can be more than one mask file for every img. Defaults to None.
        root_dir_synthesis (Path, optional): Path to synthesis masks files. Defaults to None.
        target_shape (Tuple[int, int, int], optional): Shape the images will be transformed to. 
            Defaults to None.
        connected_masks (bool, optional): Whether to use only connected masks. Defaults to False.
        dilation (int, optional): Dilation factor for masks. Defaults to 0.
        restrict_mask_to_wm (bool, optional): Whether to restrict masks to white matter regions. 
            Defaults to False.

    Raises:
        ValueError: When the amount of img files and segm files, mask folders, or mask synthesis files are not the same.
        ValueError: When dilation is greater than 0 but segmentation and mask files are not provided.
    """ 

    def __init__(
            self, 
            root_dir_img: Path, 
            root_dir_segm: Path = None, 
            root_dir_masks: Path = None, 
            root_dir_synthesis: Path = None, 
            target_shape: Tuple[int, int, int] = None, 
            connected_masks: bool = False,
            dilation: int = 0,
            restrict_mask_to_wm: bool = False):
        
        self.root_dir_img = root_dir_img  
        if(root_dir_masks):
            self.list_paths_masks = list()
            folder_list = list(root_dir_masks.glob("*")) 
            for folder in folder_list: 
                self.list_paths_masks.append(list(folder.rglob("*.nii.gz"))) 
        else:
            self.list_paths_masks = None 
        self.list_paths_segm = list(root_dir_segm.rglob("*.nii.gz")) if root_dir_segm else None 
        self.list_paths_synthesis = list(root_dir_synthesis.rglob("*.nii.gz")) if root_dir_synthesis else None 
        self.list_paths_img = list(root_dir_img.rglob("*.nii.gz"))  
        self.idx_to_element = dict() 
        self.connected_masks = connected_masks
        self.target_shape = target_shape
        self.dilation = dilation 
        self.restrict_mask_to_wm = restrict_mask_to_wm

        if(root_dir_segm and (len(self.list_paths_img) != len(self.list_paths_segm))):
            raise ValueError(f"The amount of img files and segm files must be the same. Got {len(self.list_paths_img)} and {len(self.list_paths_segm)}")        
        if(root_dir_masks and (len(self.list_paths_img)!= len(self.list_paths_masks))):
            raise ValueError(f"The amount of img files and mask folders must be the same. Got {len(self.list_paths_img)} and {len(self.list_paths_masks)}")    
        if(root_dir_synthesis and (len(self.list_paths_img)!= len(self.list_paths_synthesis))):
            raise ValueError(f"The amount of img files and mask synthesis files must be the same. Got {len(self.list_paths_img)} and {len(self.list_paths_synthesis)}") 
        if(dilation>0 and (not root_dir_segm or not root_dir_masks)):
            raise ValueError(f"For dilation the segmentation and mask files are mandatory") 

    def __len__(self) -> int: 
        return len(self.idx_to_element.keys()) 

    @abstractmethod
    def __getitem__(self, idx) -> dict:
        pass

    def _get_component_matrix(self, mask: torch.tensor, path_component_matrix: Path) -> Tuple[torch.Tensor, int]: 
        """
        Extracts connected components from mask or loads them if valid path is provided.

        Args:
            mask (torch.tensor): Mask of the mri image
            path_component_matrix (Path): Path to the component matrix
        
        Returns:
            component_matrix (torch.Tensor): Matrix containing the connected components
            n (int): Number of connected components
        """        
        
        if os.path.isfile(path_component_matrix):
            component_matrix = torch.load(path_component_matrix)
            n = torch.max(component_matrix).item()
        else:
            component_matrix, n = label(mask)
            component_matrix = torch.tensor(component_matrix)
            torch.save(component_matrix, path_component_matrix)

        return component_matrix, n

    def _get_binary_segm(self, segm: torch.tensor) -> torch.tensor:
        """
        Returns a binary mask of the white matter regions.  

        Args:
            segm (torch.tensor): Segmentation mask which can be already binary or have multiple 
            classes where class 41 and 2 are considered white matter regions

        Returns:
            binary_white_matter_segm (torch.tensor): Binary mask of the white matter regions
        """

        if segm.max() > 1:
            binary_white_matter_segm = np.logical_or(segm==41, segm==2)
        else:
            binary_white_matter_segm = segm

        return binary_white_matter_segm

    @staticmethod
    def dilate_mask(mask: torch.tensor, binary_segm: torch.tensor = None, num_pixels: int = 1, 
                    kernel_shape: str = "square") -> torch.tensor:
        """
        Dilates the input mask tensor wihtin the white matter area.

        Args:
            mask (torch.tensor): The input mask tensor.
            binary_segm (torch.tensor, optional): Binary mask of the white matter area
            num_pixels (int, optional): The number of pixels to dilate the mask by. Defaults to 1.
            kernel (str, optional): The kernel shape for dilation. Can be "square" or "cross". Defaults to "square".

        Raises:
            ValueError: When the kernel shape is not recognized.

        Returns:
            torch.tensor: The dilated mask tensor.
        """
        if kernel_shape == "square":
            kernel = torch.ones((num_pixels*2 + 1, num_pixels*2 + 1, num_pixels*2 + 1))  
        elif kernel_shape == "cross":
            kernel = torch.zeros((num_pixels*2 + 1, num_pixels*2 + 1, num_pixels*2 + 1))
            kernel[num_pixels, num_pixels, :] = 1
            kernel[num_pixels, :, num_pixels] = 1
            kernel[:, num_pixels, num_pixels] = 1
        else:
            raise ValueError(f"Kernel shape {kernel_shape} not recognized. Use 'square' or 'cross'")

        dilated_mask = F.conv3d(mask.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), 
                                padding='same'
                                ) 
        if not binary_segm is None:
            dilated_mask = dilated_mask * binary_segm.unsqueeze(0).unsqueeze(0)
        dilated_mask[dilated_mask>0]=1.
        
        #prevent shrinking of mask
        dilated_mask[mask.unsqueeze(0).unsqueeze(0).to(torch.bool)]=1.
        
        return dilated_mask.squeeze()
    
    def get_metadata(self, idx: int) -> dict:
        """
        Returns the metadata of the idx-th element in the dataset.
        
        Args:
            idx (int): Index of the element in the dataset
        
        Returns:
            metadata (dict): Metadata containing the affine matrix, header, extra information, 
                file map and data type of the original mri image
        """
        path = self.idx_to_element[idx][0] 
        img = nib.load(path) 
 
        if self.target_shape:
            img = nibabel.processing.conform(img, out_shape=self.target_shape, 
                                                 voxel_size=(1.0, 1.0, 1.0), orientation='RAS')
        metadata = {
            "affine": img.affine,
            "header": img.header,
            "extra": img.extra,
            "file_map": img.file_map,
            "dtype": img.get_data_dtype()
        }
        return metadata
    
    @staticmethod
    def postprocess(img: torch.Tensor, max_v: torch.tensor, metadata: dict) -> nib.nifti1.Nifti1Image:
        """
        Transforms the images back to their original format.  

        Args:
            img (torch.Tensor): The transformed images.
            max_v (torch.tensor): The maximum value used for scaling the images.
            metadata (dict): Metadata for creating the Nifti1Image object.

        Returns:
            nib.nifti1.Nifti1Image: The images in their original format.
        """
        # Scale to original max value
        img = (img + 1) / 2 
        img *= max_v

        img = img.permute(0, 2, 1)
        img = nib.nifti1.Nifti1Image(img.cpu().numpy(), **metadata) 

        return img

    def preprocess(
            self, 
            img: nib.nifti1.Nifti1Image, 
            masks: nib.nifti1.Nifti1Image = None, 
            segm: nib.nifti1.Nifti1Image = None, 
            synthesis_mask: nib.nifti1.Nifti1Image = None) -> Tuple[torch.Tensor, Tuple[int], torch.Tensor, torch.Tensor, torch.Tensor]: 
        """
        Preprocesses the input images and masks for the MRI dataset.
        
        Args:
            img (nib.nifti1.Nifti1Image): The MRI image.
            masks (nib.nifti1.Nifti1Image, optional): The masks image. Defaults to None.
            segm (nib.nifti1.Nifti1Image, optional): The segmentation image. Defaults to None.
            synthesis_mask (nib.nifti1.Nifti1Image, optional): The synthesis mask image. Defaults to None.
        
        Returns:
            Tuple[torch.Tensor, Tuple[int], torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the preprocessed images and additional information.
                - img (torch.Tensor): The preprocessed MRI image.
                - proc_info (Tuple[int]): Information needed for postprocessing.
                - masks (torch.Tensor): The preprocessed masks image.
                - segm (torch.Tensor): The preprocessed segmentation image.
                - synthesis_mask (torch.Tensor): The preprocessed synthesis mask image.
        """
        
        # reorient images to RAS
        if self.target_shape:
            img = nibabel.processing.conform(img, out_shape=self.target_shape, 
                                             voxel_size=(1.0, 1.0, 1.0), orientation='RAS')

        img = img.get_fdata()
        img = torch.Tensor(img)
        
        # permutation to remain backwards compatible
        img = img.permute(0, 2, 1)

        if masks != None:
            if self.target_shape:
                masks = nibabel.processing.conform(masks, out_shape=self.target_shape, 
                                                   voxel_size=(1.0, 1.0, 1.0), orientation='RAS')
            masks = masks.get_fdata() 
            # copy to avoid negative strides in numpy arrays
            masks = torch.Tensor(masks.copy())
            masks = masks.permute(0, 2, 1)
        if segm != None:
            if self.target_shape:
                segm = nibabel.processing.conform(segm, out_shape=self.target_shape, 
                                                  voxel_size=(1.0, 1.0, 1.0), orientation='RAS')
            segm = segm.get_fdata()
            # copy to avoid negative strides in numpy arrays
            segm = torch.Tensor(segm.copy())
            segm = segm.permute(0, 2, 1)
        if synthesis_mask != None:
            if self.target_shape:
                synthesis_mask = nibabel.processing.conform(synthesis_mask, 
                                                            out_shape=self.target_shape, 
                                                            voxel_size=(1.0, 1.0, 1.0), 
                                                            orientation='RAS')
            synthesis_mask = synthesis_mask.get_fdata()
            synthesis_mask = torch.Tensor(synthesis_mask.copy()) 
            synthesis_mask = synthesis_mask.permute(0, 2, 1)

        # Values below 0.01 are considered to be noise
        img[img<0.01] = 0.01 
        if masks != None:
            masks[masks<0.01] = 0
        if segm != None:
            segm[segm<0.01] = 0 
        if synthesis_mask != None:
            synthesis_mask[synthesis_mask<0.01] = 0 

        if self.dilation > 0:
            masks = DatasetMRI.dilate_mask(masks, self._get_binary_segm(segm), self.dilation)

        # Normalize the image to [-1,1] following the DDPM paper 
        max_v = torch.max(img)
        img /= max_v
        img = (img*2) - 1

        # Information needed for postprocessing 
        proc_info = (max_v) 

        return img, proc_info, masks, segm, synthesis_mask