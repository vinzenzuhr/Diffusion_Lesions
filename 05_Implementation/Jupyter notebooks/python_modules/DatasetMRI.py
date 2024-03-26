from math import floor, ceil
import nibabel as nib
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from typing import Tuple
import os
from scipy.ndimage import label

class DatasetMRI(Dataset):
    """
    Dataset for Training purposes. 
    Adapted implementation of BraTS 2023 Inpainting Challenge (https://github.com/BraTS-inpainting/2023_challenge).
    
    Contains ground truth t1n images (gt) 
    Args:
        root_dir_img: Path to img files
        root_dir_segm: Path to segmentation maps
        pad_shape: Shape the images will be transformed to

    Raises:
        UserWarning: When your input images are not (256, 256, 160)

    Returns: 
        __getitem__: Returns a dictoinary containing:
            "gt_image": Padded and cropped version of t1n 2D slice
            "segm": Segmentation of 2D slice
            "t1n_path": Path to the unpadded t1n file for this sample
            "max_v": Maximal value of t1 image (used for normalization)
            
    """

    def __init__(self, root_dir_img: Path, root_dir_segm: Path = None, root_dir_masks: Path = None, pad_shape: Tuple = (256,256,256), directDL: bool = True, seed: int = None, only_connected_masks: bool = False):
        #Initialize variables
        self.root_dir_img = root_dir_img 
        self.pad_shape = pad_shape
        self.directDL = directDL
        if(root_dir_masks):
            #make a list of lists containing all paths to masks
            self.list_paths_masks = []
            folder_list = list(root_dir_masks.glob("*")) 
            for folder in folder_list: 
                self.list_paths_masks.append(list(folder.rglob("*.nii.gz")))
        else:
            self.list_paths_masks = None
        if(root_dir_segm):
            self.list_paths_segm = list(root_dir_segm.rglob("*.nii.gz"))
        else:
            self.list_paths_segm = None 
        self.list_paths_t1n = list(root_dir_img.rglob("*.nii.gz"))
        self.idx_to_element = dict()
        self.reference_shape = (256,256,160)
        self.seed = seed  
        self.only_connected_masks = only_connected_masks

        if(root_dir_segm and (len(self.list_paths_t1n) != len(self.list_paths_segm))):
            raise ValueError(f"The amount of T1n files and segm files must be the same. Got {len(self.list_paths_t1n)} and {len(self.list_paths_segm)}")        
        if(root_dir_masks and (len(self.list_paths_t1n)!= len(self.list_paths_masks))):
            raise ValueError(f"The amount of T1n files and mask folders must be the same. Got {len(self.list_paths_t1n)} and {len(self.list_paths_masks)}")

    def __len__(self): 
        return len(self.idx_to_element.keys()) 

    def __getitem__(self, idx):
        pass

    def _get_component_matrix(self, t1n_mask, path_component_matrix): 
        """
        Extracts connected components from mask or loads them if they already exist.

        Args:
            t1n_mask (np.ndarray): Mask of the t1n image
            path_component_matrix (str): Path to the component matrix
        
        Returns:
            component_matrix (torch.Tensor): Matrix containing the connected components
            n (int): Number of connected components
        """

        # extract connected components from mask or load them if they are already exist
        if os.path.isfile(path_component_matrix):
            component_matrix = torch.load(path_component_matrix)
            n = torch.max(component_matrix).item()
        else:
            component_matrix, n = label(t1n_mask)
            component_matrix = torch.tensor(component_matrix)
            torch.save(component_matrix, path_component_matrix)

        return component_matrix, n

        
    def _get_white_matter_segm(self, segm_path: str):
        """
        Restricts the slices to white matter regions.

        Args:
            segm_path (str): Path to the segmentation file

        Returns:
            binary_white_matter_segm (np.ndarray): Binary mask of the white matter regions
        """

        t1n_segm = nib.load(segm_path)
        t1n_segm = t1n_segm.get_fdata()

        # transform segmentation if the segmentation came from Direct+DL
        if self.directDL:
            t1n_segm = np.transpose(t1n_segm)
            t1n_segm = np.flip(t1n_segm, axis=1)
        
        # restrict slices to white matter regions
        binary_white_matter_segm = np.logical_or(t1n_segm==41, t1n_segm==2)

        return binary_white_matter_segm
    
    def _get_mask(self, path_mask, path_segm):
        """
        Loads the mask and restricts it to white matter regions if a segmentation is given.

        Args:
            path_mask (str): Path to the mask file
            path_segm (str): Path to the segmentation file

        Returns:
            t1n_mask (np.ndarray): Mask of the t1n image
        """

        t1n_mask = nib.load(path_mask)
        t1n_mask = t1n_mask.get_fdata()

        # if there is a segmentation restrict mask to white matter regions
        if(self.list_paths_segm):
            binary_white_matter_segm = self._get_white_matter_segm(path_segm) 
            t1n_mask = binary_white_matter_segm * t1n_mask

        if (not t1n_mask.any()):
            print("skip t1n, because no mask inside white matter detected")
            return None
        
        return t1n_mask


    def _padding(self, t1n: torch.tensor):
        """
        Pads the images to the pad_shape. 

        Args:
            t1n (torch.Tensor): 3D t1n img

        Returns:
            t1n: The padded version of t1n.
        """

        #pad to bounding box
        size = self.pad_shape # shape of bounding box is (size,size,size)
        d, w, h = t1n.shape[-3], t1n.shape[-2], t1n.shape[-1]
        d_max, w_max, h_max = size
        d_pad = max((d_max - d) / 2, 0)
        w_pad = max((w_max - w) / 2, 0)
        h_pad = max((h_max - h) / 2, 0)
        padding = (
            int(floor(h_pad)),
            int(ceil(h_pad)),
            int(floor(w_pad)),
            int(ceil(w_pad)),
            int(floor(d_pad)),
            int(ceil(d_pad)),
        )
        t1n = F.pad(t1n, padding, value=0, mode="constant") 
        return t1n

    
    def postprocess(self, t1n: torch.Tensor, t1n_max_v: float):
        """
        Transforms the images back to their original format.
        Maps from [-1,1] to [0,1] and scales to original max value.
        
        Args:
            t1n (torch.Tensor): 3D t1n img
            t1n_max_v (float): Maximal value of t1n image (used for normalization).

        Returns:
            t1n: The padded and cropped version of t1n.
        """
        #map images from [-1,1] to [0,1]
        t1n = (t1n+1)/2
 

        #remove padding
        d, w, h = t1n.shape[-3], t1n.shape[-2], t1n.shape[-1]
        d_new, w_new, h_new = self.reference_shape # (256,256,160)
        
        d_unpad = max((d - d_new) / 2, 0)
        w_unpad = max((w - w_new) / 2, 0)
        h_unpad = max((h - h_new) / 2, 0)

        unpadding = (
            int(floor(d_unpad)),
            int(-ceil(d_unpad)) if d_unpad != 0 else None, 
            int(floor(w_unpad)),
            int(-ceil(w_unpad)) if w_unpad != 0 else None,
            int(floor(h_unpad)),
            int(-ceil(h_unpad)) if h_unpad != 0 else None,
        ) 

        t1n = t1n[..., unpadding[0]:unpadding[1], unpadding[2]:unpadding[3], unpadding[4]:unpadding[5]] 
        #scale to original max value
        t1n *= t1n_max_v
        return t1n

    def preprocess(self, t1n: np.ndarray):
        """
        Transforms the images to a more unified format.
        Normalizes to -1,1. Pad and crop to bounding box.
        
        Args:
            t1n (np.ndarray): batch of t1n from t1n file (ground truth). Shape: [B x D x W x H]

        Raises:
            UserWarning: When your input images are not (256, 256, 160)

        Returns:
            t1n: The padded and cropped version of t1n.
            t1n_max_v: Maximal value of t1n image (used for normalization).
        """

        #Size assertions
        if t1n.shape != self.reference_shape:
            raise UserWarning(f"Your t1n shape is not {self.reference_shape}, it is {t1n.shape}")

        #Normalize the image to [0,1]
        t1n[t1n<0] = 0 #Values below 0 are considered to be noise #TODO: Check validity
        t1n_max_v = np.max(t1n)
        t1n /= t1n_max_v

        #pad the image to pad_shape
        t1n = torch.Tensor(t1n)
        t1n = self._padding(t1n)

        #map images from [0,1] to [-1,1]
        t1n = (t1n*2) - 1

        return t1n, t1n_max_v
    
    def get_metadata(self, idx):
        """
        Returns the metadata of the idx-th element in the dataset.
        
        Args:
            idx (int): Index of the element in the dataset
        
        Returns:
            metadata (dict): Metadata containing the affine matrix, header, extra information, file map and data type of the original t1n image
        """
        t1n_path = self.idx_to_element[idx][0] 
        # load t1n img
        t1n_img = nib.load(t1n_path) 
        #get metadata
        metadata = {
            "affine": t1n_img.affine,
            "header": t1n_img.header,
            "extra": t1n_img.extra,
            "file_map": t1n_img.file_map,
            "dtype": t1n_img.get_data_dtype()
        }
        return metadata

    def save(self, t1n: torch.Tensor, path: Path, affine: np.ndarray, header: nib.Nifti1Header = None, extra = None, file_map = None, dtype = None):
        """
        Saves the t1n to a file.
        
        Args:
            t1n (torch.Tensor): 3D t1n img
            path (Path): Path to save the t1n file
            affine (np.ndarray): Affine matrix of the original t1n image
            header (nib.Nifti1Header): Header of the original t1n image
            extra: Extra information of the original t1n image
            file_map: File map of the original t1n image
            dtype: Data type of the original t1n image
        """  
        t1n = nib.nifti1.Nifti1Image(t1n.squeeze().numpy(), affine=affine, header=header, extra=extra, file_map=file_map, dtype=dtype)
        nib.save(t1n, path)