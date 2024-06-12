from accelerate import Accelerator
from diffusers import DiffusionPipeline
import torch 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from custom_modules import Evaluation2D, EvaluationUtils, ModelInputGenerator

class Evaluation2DSynthesis(Evaluation2D):
    """
    Class for evaluating the performance of a diffusion pipeline in 2D specific for the use case 'lesion synthesis'.
     
    Args:
        config (object): Configuration object containing evaluation settings.
        eval_dataloader (DataLoader): DataLoader for evaluation dataset.
        train_dataloader (DataLoader): DataLoader for training dataset.
        tb_summary (SummaryWriter): SummaryWriter for logging metrics.
        accelerator (Accelerator): Accelerator object for distributed training.
        model_input_generator (ModelInputGenerator): Generates the input for the model.
    """

    def __init__(self, config, eval_dataloader: DataLoader, train_dataloader: DataLoader, 
                 tb_summary: SummaryWriter, accelerator: Accelerator, model_input_generator: ModelInputGenerator):
        super().__init__(config, eval_dataloader, train_dataloader, tb_summary, accelerator, model_input_generator)

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
            self.config.add_lesion_technique, 
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
            masks (torch.tensor): The masks, which are not used.
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
            timestep=self.config.intermediate_timestep,
            generator=generator, 
            num_inference_steps = self.config.num_inference_steps,
            **parameters
        ).images     
        return synthesized_images, clean_images, synthesis_masks
     