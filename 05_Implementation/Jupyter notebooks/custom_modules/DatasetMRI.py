from math import floor, ceil
import nibabel as nib
import nibabel.processing
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
        root_dir_masks: Path to mask files. 
            Following hierarchy is expected: root_dir_masks/subject_id/*mask.nii.gz. 
            Inside subject_id folder there can be more than one mask file for every t1n file.
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

    

    def __init__(
            self, 
            root_dir_img: Path, 
            root_dir_segm: Path = None, 
            root_dir_masks: Path = None, 
            root_dir_synthesis: Path = None, 
            t1n_target_shape = None, 
            only_connected_masks: bool = False,
            dilation: int = 0,
            restrict_mask_to_wm: bool = False):
        
        #Initialize variables
        self.root_dir_img = root_dir_img  
        if(root_dir_masks):
            #make a list of lists containing all paths to masks
            self.list_paths_masks = list()
            folder_list = list(root_dir_masks.glob("*")) 
            for folder in folder_list: 
                self.list_paths_masks.append(list(folder.rglob("*.nii.gz"))) 
        else:
            self.list_paths_masks = None 

        self.list_paths_segm = list(root_dir_segm.rglob("*.nii.gz")) if root_dir_segm else None 
        self.list_paths_synthesis = list(root_dir_synthesis.rglob("*.nii.gz")) if root_dir_synthesis else None 
        self.list_paths_t1n = list(root_dir_img.rglob("*.nii.gz"))  
        self.idx_to_element = dict() 
        self.only_connected_masks = only_connected_masks
        self.t1n_target_shape = t1n_target_shape
        self.dilation = dilation 
        self.restrict_mask_to_wm = restrict_mask_to_wm

        if(root_dir_segm and (len(self.list_paths_t1n) != len(self.list_paths_segm))):
            raise ValueError(f"The amount of T1n files and segm files must be the same. Got {len(self.list_paths_t1n)} and {len(self.list_paths_segm)}")        
        if(root_dir_masks and (len(self.list_paths_t1n)!= len(self.list_paths_masks))):
            raise ValueError(f"The amount of T1n files and mask folders must be the same. Got {len(self.list_paths_t1n)} and {len(self.list_paths_masks)}")    
        if(root_dir_synthesis and (len(self.list_paths_t1n)!= len(self.list_paths_synthesis))):
            raise ValueError(f"The amount of T1n files and mask synthesis files must be the same. Got {len(self.list_paths_t1n)} and {len(self.list_paths_synthesis)}") 
        if(dilation>0 and (not root_dir_segm or not root_dir_masks)):
            raise ValueError(f"For dilation the segmentation and mask files are mandatory") 



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

    def _get_binary_segm(self, segm: torch.Tensor):
        """
        Restricts the slices to brain / white matter regions.

        Args:
            segm_path (str): Path to the segmentation file

        Returns:
            binary_white_matter_segm (np.ndarray): Binary mask of the white matter regions
        """

        # if it's not a binary segmentation restrict slices to DL+Direct white matter regions
        if segm.max() > 1:
            binary_white_matter_segm = np.logical_or(segm==41, segm==2)
        else:
            binary_white_matter_segm = segm

        return binary_white_matter_segm
    
    def _get_mask(self, path_mask, path_t1n, path_segm):
        """
        Loads the mask and restricts it to white matter regions if a segmentation is given.

        Args:
            path_mask (str): Path to the mask file
            path_segm (str): Path to the segmentation file

        Returns:
            t1n_mask (np.ndarray): Mask of the t1n image
        """

        t1n_mask = nib.load(path_mask)         
        t1n = nib.load(path_t1n)
        t1n_segm = nib.load(path_segm) if path_segm else None
        _, _, t1n_mask, t1n_segm, _ = self.preprocess(t1n = t1n, masks = t1n_mask, segm = t1n_segm)  

        # if there is a segmentation restrict mask to white matter regions
        if(self.list_paths_segm and self.restrict_mask_to_wm):
            binary_white_matter_segm = self._get_binary_segm(t1n_segm)  

            t1n_mask = binary_white_matter_segm * t1n_mask
            

        if (not t1n_mask.any()):
            print("skip t1n, because no mask inside white matter detected")
            return None
        
        return t1n_mask

    #deprecated
    @staticmethod
    def _padding(t1n: torch.tensor, pad_shape: Tuple[int, int, int]):
        print("deprecated function")
        """
        Pads the images to the pad_shape. 

        Args:
            t1n (torch.Tensor): 3D t1n img

        Returns:
            t1n: The padded version of t1n.
        """

        #pad to bounding box
        d_max, w_max, h_max = pad_shape
        d, w, h = t1n.shape[-3], t1n.shape[-2], t1n.shape[-1] 

        assert d <= d_max and w <= w_max and h <= h_max, f"The shape of the input image ({t1n.shape}) is bigger than pad_shape ({pad_shape})"

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
        t1n = F.pad(t1n, padding, value=0) 
        return t1n

    #deprecated
    @staticmethod
    def _reorient_to_ras(t1n):
        print("deprecated function")
        orientation = nib.orientations.axcodes2ornt(nib.aff2axcodes(t1n.affine))
        orientation_ras = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
        t1n_transform = nib.orientations.ornt_transform(orientation, orientation_ras)
        t1n_new = t1n.as_reoriented(t1n_transform)

        inverse_orientation = torch.tensor(nib.orientations.ornt_transform(orientation_ras, orientation))

        return t1n_new, inverse_orientation

    def dilate_mask(self, mask, binary_segm, num_pixels=1):
        kernel = torch.ones((num_pixels*2+1,num_pixels*2+1, num_pixels*2+1)) 

        dilated_mask = F.conv3d(
            mask.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0), 
            padding='same'
            )*binary_segm.unsqueeze(0).unsqueeze(0)
        dilated_mask[dilated_mask>0]=1.
        
        #prevent shrinking of mask
        dilated_mask[mask.unsqueeze(0).unsqueeze(0).to(torch.bool)]=1.
        
        return dilated_mask.squeeze()
    
    def get_metadata(self, idx):
        """
        Returns the metadata of the idx-th element in the dataset after reorienting it to RAS.
        
        Args:
            idx (int): Index of the element in the dataset
        
        Returns:
            metadata (dict): Metadata containing the affine matrix, header, extra information, file map and data type of the original t1n image
        """
        t1n_path = self.idx_to_element[idx][0] 
        # load t1n img
        t1n_img = nib.load(t1n_path) 
        if self.t1n_target_shape:
            t1n_img = nibabel.processing.conform(t1n_img, out_shape=self.t1n_target_shape, voxel_size=(1.0, 1.0, 1.0), orientation='RAS')
        #t1n_img, _ = DatasetMRI._reorient_to_ras(t1n_img)

        #get metadata
        metadata = {
            "affine": t1n_img.affine,
            "header": t1n_img.header,
            "extra": t1n_img.extra,
            "file_map": t1n_img.file_map,
            "dtype": t1n_img.get_data_dtype()
        }
        return metadata
    
    @staticmethod
    def postprocess(t1n: torch.Tensor, t1n_max_v: torch.tensor, metadata: dict): #, inverse_orientation: torch.tensor, shape_before_padding: torch.Size, shape_before_resize: torch.Size, shape_before_strip: torch.Size, metadata: dict):
        """
        Transforms the images back to their original format. 
        Stripped zero slices can not be recovered.

        """
        #scale to original max value
        t1n = (t1n+1)/2 
        t1n *= t1n_max_v

        #remove padding
        #d, h = t1n.shape[-3], t1n.shape[-1]
        #d_new, h_new = shape_before_padding[-3], shape_before_padding[-1]
        
        #d_unpad = max((d - d_new) / 2, 0) 
        #h_unpad = max((h - h_new) / 2, 0)

        #unpadding = (
        #    int(floor(d_unpad)),
        #    int(-ceil(d_unpad)) if d_unpad != 0 else None,  
        #    int(floor(h_unpad)),
        #    int(-ceil(h_unpad)) if h_unpad != 0 else None,
        #) 

        #t1n = t1n[..., unpadding[0]:unpadding[1], :, unpadding[2]:unpadding[3]] 
        
        #resize to original shape 
        #t1n = torch.nn.functional.interpolate(t1n.unsqueeze(0).unsqueeze(0), size=tuple(shape_before_resize), mode='trilinear').squeeze()

        # permutation to remain backwards compatible
        t1n = t1n.permute(0, 2, 1)

        #reorient image to original orientation
        t1n = nib.nifti1.Nifti1Image(t1n.cpu().numpy(), **metadata) 
        #t1n = t1n.as_reoriented(inverse_orientation.squeeze().cpu().numpy())

        return t1n
    
    #deprecated
    @staticmethod
    def strip(t1n, masks = None, segm = None, synthesis_mask = None): 
        print("deprecated function")
        while t1n[:,:,0].abs().sum() == 0:
            t1n = t1n[:,:,1:]
            masks = masks[:,:,1:] if masks != None else None
            segm = segm[:,:,1:] if segm != None else None
            synthesis_mask = synthesis_mask[:,:,1:] if synthesis_mask != None else None

        while t1n[:,:,-1].abs().sum() == 0: 
            t1n = t1n[:,:,:-1]
            masks = masks[:,:,:-1] if masks != None else None
            segm = segm[:,:,:-1] if segm != None else None
            synthesis_mask = synthesis_mask[:,:,:-1] if synthesis_mask!= None else None
            
        while t1n[:,0,:].abs().sum() == 0:
            t1n = t1n[:,1:,:]
            masks = masks[:,1:,:] if masks != None else None
            segm = segm[:,1:,:] if segm != None else None
            synthesis_mask = synthesis_mask[:,1:,:] if synthesis_mask != None else None

        while t1n[:,-1,:].abs().sum() == 0:
            t1n = t1n[:,:-1,:]
            masks = masks[:,:-1,:] if masks != None else None
            segm = segm[:,:-1,:] if segm != None else None
            synthesis_mask = synthesis_mask[:,:-1,:] if synthesis_mask != None else None
            
        while t1n[0,:,:].abs().sum() == 0:
            t1n = t1n[1:,:,:]
            masks = masks[1:,:,:] if masks != None else None
            segm = segm[1:,:,:] if segm != None else None
            synthesis_mask = synthesis_mask[1:,:,:] if synthesis_mask != None else None

        while t1n[-1,:,:].abs().sum() == 0:
            t1n = t1n[:-1,:,:] 
            masks = masks[:-1,:,:] if masks != None else None
            segm = segm[:-1,:,:] if segm != None else None
            synthesis_mask = synthesis_mask[:-1,:,:] if synthesis_mask != None else None
        return t1n, masks, segm, synthesis_mask


    def preprocess(
            self, 
            t1n: nib.nifti1.Nifti1Image, 
            masks: nib.nifti1.Nifti1Image = None, 
            segm: nib.nifti1.Nifti1Image = None, 
            synthesis_mask: nib.nifti1.Nifti1Image = None): 
         
        # reorient images to RAS
        #t1n, inverse_orientation = DatasetMRI._reorient_to_ras(t1n)
        if self.t1n_target_shape:
            t1n = nibabel.processing.conform(t1n, out_shape=self.t1n_target_shape, voxel_size=(1.0, 1.0, 1.0), orientation='RAS')

        t1n = t1n.get_fdata()
        t1n = torch.Tensor(t1n)
        
        # permutation to remain backwards compatible
        t1n = t1n.permute(0, 2, 1)

        if masks != None:
            #masks, _ = DatasetMRI._reorient_to_ras(masks)
            if self.t1n_target_shape:
                masks = nibabel.processing.conform(masks, out_shape=self.t1n_target_shape, voxel_size=(1.0, 1.0, 1.0), orientation='RAS')
            masks = masks.get_fdata() 
            # copy to avoid negative strides in numpy arrays
            masks = torch.Tensor(masks.copy())
            masks = masks.permute(0, 2, 1)
        if segm != None:
            #segm, _ = DatasetMRI._reorient_to_ras(segm)
            if self.t1n_target_shape:
                segm = nibabel.processing.conform(segm, out_shape=self.t1n_target_shape, voxel_size=(1.0, 1.0, 1.0), orientation='RAS')
            segm = segm.get_fdata()
            segm = torch.Tensor(segm.copy())
            segm = segm.permute(0, 2, 1)
        if synthesis_mask != None:
            if self.t1n_target_shape:
                synthesis_mask = nibabel.processing.conform(synthesis_mask, out_shape=self.t1n_target_shape, voxel_size=(1.0, 1.0, 1.0), orientation='RAS')
            #synthesis_mask, _ = DatasetMRI._reorient_to_ras(synthesis_mask)
            synthesis_mask = synthesis_mask.get_fdata()
            synthesis_mask = torch.Tensor(synthesis_mask.copy()) 
            synthesis_mask = synthesis_mask.permute(0, 2, 1)

        # remove all zero slices from t1n
        t1n[t1n<0.01] = 0.01 #Values below 0.01 are considered to be noise
        if masks != None:
            masks[masks<0.01] = 0
        if segm != None:
            segm[segm<0.01] = 0 
        if synthesis_mask != None:
            synthesis_mask[synthesis_mask<0.01] = 0 

        if self.dilation > 0:
            masks = self.dilate_mask(masks, self._get_binary_segm(segm), self.dilation)
        


        
        #shape_before_strip = t1n.shape
        #t1n, masks, segm, synthesis_mask = DatasetMRI.strip(t1n, masks, segm, synthesis_mask)

        # resize image to the max, which fits inside target_shape
        #shape_before_resize=t1n.shape
        #t1n = self.resize_image(t1n)
        
        #pad the image to target_shape
        #shape_before_padding = t1n.shape
        #pad_shape = (self.img_target_shape[0], t1n.shape[1], self.img_target_shape[1])
        #t1n = DatasetMRI._padding(t1n, pad_shape)

        #Normalize the image to [-1,1] following the DDPM paper 
        t1n_max_v = torch.max(t1n)
        t1n /= t1n_max_v # [0,1]
        t1n = (t1n*2) - 1

        #information needed for postprocessing
        #proc_info = (t1n_max_v, inverse_orientation, shape_before_padding, shape_before_resize, shape_before_strip)
        proc_info = (t1n_max_v)

        #if masks != None:
        #    masks = self.resize_image(masks)
        #    masks = DatasetMRI._padding(masks, pad_shape)
        #if segm != None:
        #    segm = self.resize_image(segm)
        #    segm = DatasetMRI._padding(segm, pad_shape)
        #if synthesis_mask != None:
        #    synthesis_mask = self.resize_image(synthesis_mask)
        #    synthesis_mask = DatasetMRI._padding(synthesis_mask, pad_shape)

        return t1n, proc_info, masks, segm, synthesis_mask

    #deprecated
    def resize_image(self, img):
        print("deprecated function")
        img_shape = torch.tensor([img.shape[-3], img.shape[-1]])
        shape_max = torch.tensor(self.img_target_shape)
        factors = shape_max / img_shape
        img = torch.nn.functional.interpolate(img.unsqueeze(0).unsqueeze(0), scale_factor=factors.min().item()).squeeze()
        return img

    @staticmethod
    def save(t1n: torch.Tensor, path: Path):
        """
        Saves the t1n to a file.
        
        Args:
            t1n (torch.Tensor): 3D t1n img
            path (Path): Path to save the t1n file 
        """
        nib.save(t1n, path)