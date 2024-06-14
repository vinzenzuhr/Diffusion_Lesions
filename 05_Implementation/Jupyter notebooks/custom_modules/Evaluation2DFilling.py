
from accelerate import Accelerator
from diffusers import DiffusionPipeline
import torch 
from torch.utils.data import DataLoader 

from custom_modules import Evaluation2D, ModelInputGenerator, Logger

class Evaluation2DFilling(Evaluation2D):
    """
    Class for evaluating the performance of a diffusion pipeline in 2D specific for the use case 'lesion filling'.

    Args:
        eval_dataloader (DataLoader): The data loader for evaluation data.
        train_dataloader (DataLoader): The data loader for training data.
        logger (Logger): Object for logging.
        accelerator (Accelerator): The accelerator for distributed training.
        num_inference_steps (int): The number of inference steps.
        model_input_generator (ModelInputGenerator): ModelInputGenerator object for generating different model inputs.
        output_dir (str): The output directory for saving results.
        eval_loss_timesteps (list[int]): List of timesteps to evalute validation loss.
        evaluate_num_batches (int, optional): The number of batches to evaluate. Defaults to -1 (evaluate all batches).
        seed (int, optional): The random seed for reproducibility. Defaults to None.
    """

    def __init__(self, eval_dataloader: DataLoader, train_dataloader: DataLoader, 
                 logger: Logger, accelerator: Accelerator, num_inference_steps: int, 
                 model_input_generator: ModelInputGenerator, output_dir: str, eval_loss_timesteps: list[int], 
                 evaluate_num_batches: int = -1, seed: int = None):
        super().__init__(eval_dataloader, train_dataloader, logger, accelerator, num_inference_steps, model_input_generator, output_dir, 
                         eval_loss_timesteps, evaluate_num_batches, seed)
    
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
                        generator: torch.Generator, parameters: dict = {}) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
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
            num_inference_steps=self.num_inference_steps,
            **parameters
        ).images 
        
        return inpainted_images, clean_images, masks