
from accelerate import Accelerator
from diffusers import DiffusionPipeline
import torch 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from custom_modules import Evaluation2D

class Evaluation2DFilling(Evaluation2D):
    """
    Class for evaluating the performance of a diffusion pipeline in 2D specific for the use case 'lesion filling'.

    Args:
        config (object): Configuration object containing evaluation settings.
        eval_dataloader (DataLoader): DataLoader for evaluation dataset.
        train_dataloader (DataLoader): DataLoader for training dataset.
        tb_summary (SummaryWriter): SummaryWriter for logging metrics.
        accelerator (Accelerator): Accelerator object for distributed training.
    """

    def __init__(self, config, eval_dataloader: DataLoader, train_dataloader: DataLoader, 
                 tb_summary: SummaryWriter, accelerator: Accelerator):
        super().__init__(config, eval_dataloader, train_dataloader, tb_summary, accelerator) 
    
    def _get_image_lists(self, images: torch.tensor, clean_images: torch.tensor, masks: torch.tensor, 
                         batch: torch.tensor) -> tuple[list[torch.tensor, torch.tensor, torch.tensor, torch.tensor], list[str, str, str, str]]:
        """
        Get a list of the images and their corresponding titles.

        Args:
            images (torch.tensor): The processed images.
            clean_images (torch.tensor): The ground truth images.
            masks (torch.tensor): The masks.
            batch (torch.tensor): The batch with additional data.

        Returns:
            tuple: A tuple containing two lists. The first list contains the images and the second list contains their titles.
        """
        masked_images = clean_images*(1-masks)
        
        # change range from [-1,1] to [0,1]
        images = (images+1) / 2
        masked_images = (masked_images+1) / 2
        clean_images = (clean_images+1) / 2  

        list = [images, masked_images, clean_images, masks]
        title_list = ["images", "masked_images", "clean_images", "masks"] 
        return list, title_list

    def _start_pipeline(self, pipeline: DiffusionPipeline, batch: torch.tensor, 
                        generator: torch.Generator, parameters: dict = {}):
        """
        Starts the pipeline for inpainting voided images.

        Args:
            pipeline (DiffusionPipeline): The diffusion pipeline used for inpainting.
            batch (torch.tensor): Batch with data.
            generator (torch.Generator): The random number generator.
            parameters (dict, optional): Additional parameters for the pipeline. Defaults to {}.

        Returns:
            tuple: A tuple containing the inpainted images, ground truth images, and masks.
        """
        clean_images = batch["gt_image"]
        masks = batch["mask"] 
        voided_images = clean_images * (1-masks)
         
        inpainted_images = pipeline(
            voided_images,
            masks,
            generator=generator, 
            num_inference_steps=self.config.num_inference_steps,
            **parameters
        ).images 
        
        return inpainted_images, clean_images, masks