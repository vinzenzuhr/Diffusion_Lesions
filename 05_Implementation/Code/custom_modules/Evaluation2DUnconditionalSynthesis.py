from accelerate import Accelerator
from diffusers import DiffusionPipeline
import torch 
from torch.utils.data import DataLoader 

from custom_modules import Evaluation2D, EvaluationUtils, ModelInputGenerator, Logger

class Evaluation2DUnconditionalSynthesis(Evaluation2D):
    """
    Class for evaluating the performance of a diffusion pipeline in 2D specific for the use case 'unconditional lesion synthesis'.

    Args:
        intermediate_timestep (int): The intermediate timestep to start the diffusion process.
        add_lesion_technique (str): The technique used to add the coarse lesions to the images.
        eval_dataloader (DataLoader): The dataloader for evaluation data.
        train_dataloader (DataLoader): The dataloader for training data.
        logger (Logger): Object for logging.
        accelerator (Accelerator): Accelerator object for distributed training.
        num_inference_steps (int): Number of inference steps.
        model_input_generator (ModelInputGenerator): ModelInputGenerator object for generating different model inputs.
        output_dir (str): Output directory for saving results.
        eval_loss_timesteps (list[int]): List of timesteps to evalute validation loss.
        evaluate_num_batches (int, optional): Number of batches to evaluate. Defaults to -1 (all batches).
        seed (int, optional): Random seed. Defaults to None.
    """

    def __init__(self, intermediate_timestep: int, add_lesion_technique: str, eval_dataloader: DataLoader, train_dataloader: DataLoader, 
                 logger: Logger, accelerator: Accelerator, num_inference_steps: int, model_input_generator: ModelInputGenerator, output_dir: str,
                 eval_loss_timesteps: list[int], evaluate_num_batches: int = -1, seed: int = None):
        super().__init__(eval_dataloader, train_dataloader, logger, accelerator, num_inference_steps, model_input_generator, output_dir, 
                         eval_loss_timesteps, evaluate_num_batches, seed)
        self.intermediate_timestep = intermediate_timestep
        self.add_lesion_technique = add_lesion_technique

    def _add_coarse_lesions(self, clean_images: torch.tensor, batch: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        """
        Adds coarse lesions to the clean images based on the synthesis masks.

        Args:
            clean_images (torch.tensor): The original images to add lesions to.
            batch (torch.tensor): The batch containing the synthesis masks and masks.

        Returns:
            tuple[torch.tensor, torch.tensor]: A tuple containing the images with lesions and the synthesis masks.
        """
        synthesis_masks = batch["synthesis"] 
        masks = batch["mask"].to(torch.bool) 
        lesion_intensity = EvaluationUtils.get_lesion_intensity(
            self.add_lesion_technique, 
            clean_images[masks])
        
        images_with_lesions = clean_images.clone()
        images_with_lesions[synthesis_masks.to(torch.bool)] = lesion_intensity 
        return images_with_lesions, synthesis_masks

    def _get_image_lists(self, images: torch.tensor, clean_images: torch.tensor, masks: torch.tensor, 
                         batch: torch.tensor) -> tuple[list[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor], list[str, str, str, str, str]]:
        """
        Get a list of the images and their corresponding titles.

        Args:
            images (torch.tensor): The processed images.
            clean_images (torch.tensor): The ground truth images.
            masks (torch.tensor): The masks for the images.
            batch (torch.tensor): The batch with additional data.

        Returns:
            tuple: A tuple containing two lists. The first list contains the images and the second list contains their titles.
        """
        images_with_lesions, synthesis_masks = self._add_coarse_lesions(clean_images, batch)
        masked_images = clean_images * (1-synthesis_masks)
        
        # change range from [-1,1] to [0,1]
        images = (images+1) / 2
        masked_images = (masked_images+1) / 2
        clean_images = (clean_images+1) / 2
        images_with_lesions = (images_with_lesions+1) / 2 

        list = [images, masked_images, clean_images, synthesis_masks, images_with_lesions]
        title_list = ["images", "masked_images", "clean_images", "synthesis_masks", "images_with_lesions"] 
        return list, title_list

    def _start_pipeline(self, pipeline: DiffusionPipeline, batch: torch.tensor, generator: torch.Generator, 
                        parameters: dict = {}) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Starts the pipeline for synthesizing images with lesions.

        Args:
            pipeline (DiffusionPipeline): The diffusion pipeline used for inpainting.
            batch (torch.tensor): Batch with data.
            generator (torch.Generator): The random number generator.
            parameters (dict, optional): Additional parameters for the pipeline. Defaults to {}.

        Returns:
            tuple: A tuple containing the synthesized images, ground truth images, and masks with lesions to be synthesized.
        """
        assert (type(pipeline).__name__ == "GuidedRePaintPipeline" 
                or type(pipeline).__name__ == "GuidedPipelineUnconditional"
                or type(pipeline).__name__ == "GuidedPipelineConditional") , \
            "Pipeline must be of type GuidedPipelineUnconditional, GuidedPipelineConditional or GuidedRePaintPipeline"

        clean_images = batch["gt_image"]
        images_with_lesions, synthesis_masks = self._add_coarse_lesions(clean_images, batch)
 
        synthesized_images = pipeline(
            images_with_lesions,
            synthesis_masks,
            timestep=self.intermediate_timestep,
            generator=generator, 
            num_inference_steps = self.config.num_inference_steps,
            **parameters
        ).images     
        return synthesized_images, clean_images, synthesis_masks
     