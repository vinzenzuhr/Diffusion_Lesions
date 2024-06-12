from typing import Union, Callable
 
import diffusers  
from diffusers import DDIMScheduler, DDPMScheduler, DiffusionPipeline
import torch  
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler  

 
from custom_modules import Training, DatasetMRI2D, DatasetMRI3D, Evaluation2D, Evaluation3D
import pseudo3D

class TrainingConditional(Training):
    """
    Class to train conditional diffusion models.  

    
    Args:
        config (object): The configuration object containing various training parameters.
        model (Union[diffusers.UNet2DModel, pseudo3D.UNet2DModel]): The conditional model to be trained.
        noise_scheduler (Union[DDIMScheduler, DDPMScheduler]): The noise scheduler used for adding noise to the images during training.
        optimizer (Optimizer): The optimizer used for updating the model parameters.
        lr_scheduler (LRScheduler): The learning rate scheduler used for adjusting the learning rate during training.
        datasetTrain (DatasetMRI2D): The training dataset.
        datasetEvaluation (DatasetMRI2D): The evaluation dataset for 2D images.
        dataset3DEvaluation (DatasetMRI3D): The evaluation dataset for 3D images.
        evaluation2D (Evaluation2D): The evaluation object for 2D images, used for computing metrics.
        evaluation3D (Evaluation3D): The evaluation object for 3D images, used for computing metrics.
        pipelineFactory (Callable[[Union[diffusers.UNet2DModel, pseudo3D.UNet2DModel], Union[DDIMScheduler, DDPMScheduler]], DiffusionPipeline]): 
            A callable that creates a diffusion pipeline given the model and noise scheduler.
        sorted_slice_sample_size (int): The number of sorted slices within one sample from the Dataset. 
            Defaults to 1. This is needed for the pseudo3Dmodels, where the model expects that 
            the slices within one batch are next to each other in the 3D volume.
        min_snr_loss (bool): A boolean indicating whether to use the minimum SNR loss. Default is False.

    """

    def __init__(
            self, 
            config, 
            model: Union[diffusers.UNet2DModel, pseudo3D.UNet2DModel], 
            noise_scheduler: Union[DDIMScheduler, DDPMScheduler], 
            optimizer: Optimizer, 
            lr_scheduler: LRScheduler, 
            datasetTrain: DatasetMRI2D, 
            datasetEvaluation: DatasetMRI2D, 
            dataset3DEvaluation: DatasetMRI3D, 
            evaluation2D: Evaluation2D, 
            evaluation3D: Evaluation3D, 
            pipelineFactory: Callable[[Union[diffusers.UNet2DModel, pseudo3D.UNet2DModel], Union[DDIMScheduler, DDPMScheduler]], DiffusionPipeline], 
            sorted_slice_sample_size: int = 1, 
            min_snr_loss: bool = False, 
            ):
        super().__init__(config, model, noise_scheduler, optimizer, lr_scheduler, datasetTrain, 
                         datasetEvaluation, dataset3DEvaluation, evaluation2D, evaluation3D, 
                         pipelineFactory, sorted_slice_sample_size, min_snr_loss)

    def _get_training_input(self, batch: torch.tensor, generator: torch.Generator = None, 
                            timesteps: torch.tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the training input for the model.

        The input for the model consists of the noisy images and the concatenation of the voided images and masks to be conditioned.

        Args:
            batch (torch.tensor): The input batch of data.
            generator (torch.Generator, optional): The random number generator. Defaults to None.
            timesteps (torch.tensor, optional): Predefined timesteps for diffusing each image. 
                Defaults to None.

        Returns:
            A tuple containing the input, noise, and timesteps.

        """
        clean_images = batch["gt_image"] 
        masks = batch["mask"] 
        noisy_images, noise, timesteps = self._get_noisy_images(clean_images, generator, timesteps)

        voided_images = clean_images*(1-masks)
 
        input=torch.cat((noisy_images, voided_images, masks), dim=1)
        return input, noise, timesteps

    def evaluate(self, pipeline: DiffusionPipeline = None, deactivate_save_model: bool = False):
        """
        Evaluates the model on the evaluation datasets. 

        Args:
            pipeline: The diffusion pipeline.
            deactivate_save_model: A flag indicating whether to deactivate saving the model during evaluation.

        """
        self.model.eval()
        # Create pipeline if not given
        if pipeline is None:
            pipeline = self.pipelineFactory(self.accelerator.unwrap_model(self.model), self.noise_scheduler) 
        pipeline = self.accelerator.prepare(pipeline)
        pipeline.to(self.accelerator.device)
        
        # Evaluate 2D images
        if not self.config.deactivate2Devaluation and ((self.epoch) % self.config.evaluate_epochs == 0 or self.epoch == self.config.num_epochs - 1): 
            self.evaluation2D.evaluate(
                pipeline, 
                self.global_step, 
                self._get_training_input, 
                parameters = {},
                deactivate_save_model=deactivate_save_model)

        # Evaluate 3D images composed of 2D slices
        if (not self.config.deactivate3Devaluation and ((self.epoch) % self.config.evaluate_3D_epochs == 0 or self.epoch == self.config.num_epochs - 1)): 
            self.evaluation3D.evaluate(pipeline, self.global_step, 
                parameters = {})



 