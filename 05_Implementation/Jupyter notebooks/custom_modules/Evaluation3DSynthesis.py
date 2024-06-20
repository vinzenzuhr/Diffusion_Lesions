
import math

from accelerate import Accelerator 
from diffusers import DiffusionPipeline
import nibabel as nib
import torch 
from torch.utils.data import DataLoader 

from custom_modules import Evaluation3D,  EvaluationUtils , Logger

class Evaluation3DSynthesis(Evaluation3D):
    """
    Evaluation class for 3D synthesis.

    Args:
        intermediate_timestep (int): The intermediate timestep to start the diffusion process.
        add_lesion_technique (str): The technique used to add the coarse lesions to the images.
        dataloader (DataLoader): DataLoader object for loading 3D evaluation data.
        logger (Logger): The logger object for logging.
        accelerator (Accelerator): The accelerator object for distributed training.
        output_dir (str): The output directory for saving results.
        num_inference_steps (int): The number of inference steps.
        eval_batch_size (int): The evaluation batch size.
        sorted_slice_sample_size (int, optional):  The number of sorted slices within one sample from the Dataset. 
            Defaults to 1. This is needed for the pseudo3Dmodels, where the model expects that 
            the slices within one batch are next to each other in the 3D volume.
        evaluate_num_batches (int, optional): Number of batches to evaluate. Defaults to -1 (all batches). 
        seed (int, optional): Seed for random number generation. Defaults to None.
    """

    def __init__(self, intermediate_timestep: int, add_lesion_technique: str, dataloader: DataLoader, 
                 logger: Logger, accelerator: Accelerator, output_dir: str, filename: str, num_inference_steps: int, 
                 eval_batch_size: int, sorted_slice_sample_size: int = 1, evaluate_num_batches: int = -1, 
                 seed: int = None,):
        super().__init__(dataloader, logger, accelerator, output_dir, filename, evaluate_num_batches) 
        self.intermediate_timestep = intermediate_timestep
        self.add_lesion_technique = add_lesion_technique
        self.num_inference_steps = num_inference_steps
        self.eval_batch_size = eval_batch_size
        self.sorted_slice_sample_size = sorted_slice_sample_size
        self.seed = seed 

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
            self.add_lesion_technique, 
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
                if position_in_package == self.sorted_slice_sample_size:
                    position_in_package = 0  
        slice_indices = torch.cat(slice_indices, 0)
    
        # Create chunks of slices which have to be synthesized 
        stacked_images = torch.stack((images_with_lesions[:, :, slice_indices, :], synthesis_masks[:, :, slice_indices, :]), dim=0)
        stacked_images = stacked_images.permute(0, 3, 1, 2, 4)
        chunk_size = self.eval_batch_size if self.sorted_slice_sample_size == 1 else self.sorted_slice_sample_size
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
                timestep=self.intermediate_timestep,
                num_inference_steps = self.num_inference_steps,
                generator=[torch.Generator(device=self.accelerator.device).manual_seed(self.seed) for _ in range(chunk_images.shape[0])],
                **parameters
            ).images 
            images.append(new_images)
        images = torch.cat(images, dim=0) 
        images = images.permute(1, 2, 0, 3) 
                
        return images, clean_images, slice_indices, synthesis_masks