
import math

from accelerate import Accelerator 
from diffusers import DiffusionPipeline
import nibabel as nib
import torch 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 

from custom_modules import Evaluation3D,  EvaluationUtils 

class Evaluation3DSynthesis(Evaluation3D):
    """
    Class for evaluating the performance of a diffusion pipeline with 3D images for the use case 'lesion synthesis'. 

    Args:
        config (object): Configuration object containing evaluation settings.
        dataloader (DataLoader): DataLoader object for loading 3D evaluation data.
        tb_summary (SummaryWriter): SummaryWriter for logging metrics.
        accelerator (Accelerator): Accelerator object for distributed training. 
    """
    
    def __init__(self, config, dataloader: DataLoader, tb_summary: SummaryWriter, accelerator: Accelerator):
        super().__init__(config, dataloader, tb_summary, accelerator) 

    def _start_pipeline(self, pipeline: DiffusionPipeline, batch: torch.tensor, sample_idx: torch.tensor, 
                        parameters: dict = {}) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Start the synthesis pipeline for a given batch and sample.

        Args:
            pipeline (DiffusionPipeline): Diffusion pipeline object.
            batch (torch.tensor): Input batch of data.
            sample_idx (int): Index of the sample in the batch.
            parameters (dict, optional): Additional parameters for the pipeline. Defaults to {}.

        Returns:
            tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]: Tuple containing the inpainted images,
                clean images, slice indices, and synthesis masks.
        """ 
        clean_images = batch["gt_image"][sample_idx]  
        synthesis_masks = batch["synthesis"][sample_idx]   
        masks = batch["mask"][sample_idx].to(torch.bool)  

        # add coarse lesions to the clean images based on the synthesis masks
        lesion_intensity = EvaluationUtils.get_lesion_intensity(
            self.config.add_lesion_technique, 
            clean_images[masks])
        images_with_lesions = clean_images.clone()
        images_with_lesions[synthesis_masks.to(torch.bool)] = lesion_intensity

        # Get slices which have to be synthesized along the horizontal section.
        # Make sure there are packages of sorted_slice_sample_size slices, which are next to 
        # each other in the 3D volume.
        position_in_package = 0
        slice_indices = []
        for slice_idx in torch.arange(clean_images.shape[2]):
            if (synthesis_masks[:, :, slice_idx, :]).any() or position_in_package>0:
                slice_indices.append(slice_idx.unsqueeze(0)) 
                position_in_package += 1
                if position_in_package == self.config.sorted_slice_sample_size:
                    position_in_package = 0  
        slice_indices = torch.cat(slice_indices, 0)
    
        # Create chunks of slices which have to be synthesized 
        stacked_images = torch.stack((images_with_lesions[:, :, slice_indices, :], synthesis_masks[:, :, slice_indices, :]), dim=0)
        stacked_images = stacked_images.permute(0, 3, 1, 2, 4)
        chunk_size = self.config.eval_batch_size if self.config.sorted_slice_sample_size == 1 else self.config.sorted_slice_sample_size
        chunks = torch.chunk(stacked_images, math.ceil(stacked_images.shape[1]/chunk_size), dim=1)

        # Synthesize all slices 
        images = [] 
        for chunk in chunks:  
            chunk_images = chunk[0]
            chunk_masks = chunk[1] 

            # use the same generator for every picture to create a more homogeneous output
            new_images = pipeline(
                chunk_images,
                chunk_masks,
                timestep=self.config.intermediate_timestep,
                num_inference_steps = self.config.num_inference_steps,
                generator=[torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed) for _ in range(chunk_images.shape[0])],
                **parameters
            ).images 
            images.append(new_images)
        images = torch.cat(images, dim=0) 
        images = images.permute(1, 2, 0, 3) 
                
        return images, clean_images, slice_indices, synthesis_masks
    
    def _save_image(self, final_3d_image: nib.nifti1.Nifti1Image, save_dir: str): 
        """
        Saves the final 3D image to disk.
        
        Args:
            final_3d_image (nib.nifti1.Nifti1Image): The final 3D nifti image to be saved.
            save_dir (str): The directory where the image should be saved.
        """
        nib.save(final_3d_image, f"{save_dir}/FLAIR.nii.gz")