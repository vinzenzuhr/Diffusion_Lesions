import math

from accelerate import Accelerator 
from diffusers import DiffusionPipeline
import nibabel as nib
import torch 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 

from custom_modules import Evaluation3D

class Evaluation3DFilling(Evaluation3D):
    """
    Class for evaluating the performance of a diffusion pipeline with 3D images for the use case 'lesion filling'. 

    Args:
        config (object): Configuration object containing evaluation settings.
        dataloader (DataLoader): DataLoader object for loading 3D evaluation data.
        tb_summary (SummaryWriter): SummaryWriter for logging metrics.
        accelerator (Accelerator): Accelerator object for distributed training. 
    """

    def __init__(self, config, dataloader: DataLoader, tb_summary: SummaryWriter, accelerator: Accelerator):
        super().__init__(config, dataloader, tb_summary, accelerator)

    def _start_pipeline(self, pipeline: DiffusionPipeline, batch: torch.tensor, sample_idx: int, 
                        parameters: dict = {}) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Start the image inpainting pipeline for a given batch and sample.

        Args:
            pipeline (DiffusionPipeline): Inpainting pipeline object.
            batch (torch.tensor): Input batch of images and masks.
            sample_idx (int): Index of the sample in the batch.
            parameters (dict, optional): Additional parameters for the pipeline. Defaults to {}.

        Returns:
            tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]: Tuple containing the inpainted images,
                clean images, slice indices, and masks.
        """

        clean_images = batch["gt_image"][sample_idx] 
        masks = batch["mask"][sample_idx] 
    
        # Get slices which have to be inpainted along the horizontal section.
        # Make sure there are packages of sorted_slice_sample_size slices, which are next to 
        # each other in the 3D volume.
        position_in_package = 0
        slice_indices = []
        for slice_idx in torch.arange(clean_images.shape[2]):
            if masks[:, :, slice_idx, :].any() or position_in_package > 0:
                slice_indices.append(slice_idx.unsqueeze(0)) 
                position_in_package += 1
                if position_in_package == self.config.sorted_slice_sample_size:
                    position_in_package = 0  
        slice_indices = torch.cat(slice_indices, 0)
         
        # Create chunks of slices which have to be inpainted 
        stacked_images = torch.stack((clean_images[:, :, slice_indices, :], masks[:, :, slice_indices, :]), dim=0)
        stacked_images = stacked_images.permute(0, 3, 1, 2, 4) 
        chunk_size = self.config.eval_batch_size if self.config.sorted_slice_sample_size == 1 else self.config.sorted_slice_sample_size
        chunks = torch.chunk(stacked_images, math.ceil(stacked_images.shape[1]/chunk_size), dim=1)
         

        # Inpaint all slices with a seed per 3D image
        images = []  
        generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed)
        for chunk in chunks:
            chunk_images = chunk[0]
            chunk_masks = chunk[1]
            chunk_voided_images = chunk_images * (1-chunk_masks)

            new_images = pipeline(
                chunk_voided_images,
                chunk_masks,
                num_inference_steps=self.config.num_inference_steps,
                generator=generator,
                **parameters
            ).images 
            images.append(new_images)
        images = torch.cat(images, dim=0) 
        images = images.permute(1, 2, 0, 3) 

        return images, clean_images, slice_indices, masks
    
    def _save_image(self, final_3d_image: nib.nifti1.Nifti1Image, save_dir: str): 
        """
        Save the final 3D nifti image to a specified directory.

        Args:
            final_3d_images (nib.nifti1.Nifti1Image): Final 3D images to be saved.
            save_dir (str): Directory path to save the images.
        """
        nib.save(final_3d_image, f"{save_dir}/T1.nii.gz")