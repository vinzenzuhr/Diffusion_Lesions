from typing import Union, Callable
 
import diffusers  
from diffusers import DDIMScheduler, DDPMScheduler, DiffusionPipeline
import torch  
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler  

 
from custom_modules import Training, DatasetMRI2D, DatasetMRI3D, Evaluation2D, Evaluation3D, ModelInputGenerator
from . import pseudo3D

class TrainingConditional(Training):
    """
    Class to train conditional diffusion models.  

    
    Args:
        config (object): The configuration object containing various training parameters.
        model (Union[diffusers.UNet2DModel, pseudo3D.UNet2DModel]): The conditional model to be trained.
        noise_scheduler (Union[DDIMScheduler, DDPMScheduler]): The noise scheduler used for adding noise to the images during training.
        optimizer (Optimizer): The optimizer used for updating the model parameters.
        lr_scheduler (LRScheduler): The learning rate scheduler used for adjusting the learning rate during training.
        dataset_train (DatasetMRI2D): The training dataset.
        dataset_evaluation (DatasetMRI2D): The evaluation dataset for 2D images.
        dataset_3D_evaluation (DatasetMRI3D): The evaluation dataset for 3D images.
        evaluation2D (Evaluation2D): The evaluation object for 2D images, used for computing metrics.
        evaluation3D (Evaluation3D): The evaluation object for 3D images, used for computing metrics.
        pipeline_factory (Callable[[Union[diffusers.UNet2DModel, pseudo3D.UNet2DModel], Union[DDIMScheduler, DDPMScheduler]], DiffusionPipeline]): 
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
            dataset_train: DatasetMRI2D, 
            dataset_evaluation: DatasetMRI2D, 
            dataset_3D_evaluation: DatasetMRI3D, 
            Evaluation2D: Evaluation2D, 
            Evaluation3D: Evaluation3D, 
            pipeline_factory: Callable[[Union[diffusers.UNet2DModel, pseudo3D.UNet2DModel], Union[DDIMScheduler, DDPMScheduler]], DiffusionPipeline], 
            sorted_slice_sample_size: int = 1, 
            min_snr_loss: bool = False, 
            ): 
        super().__init__(config, model, noise_scheduler, optimizer, lr_scheduler, dataset_train, 
                         dataset_evaluation, dataset_3D_evaluation, pipeline_factory, 
                         sorted_slice_sample_size, min_snr_loss) 
        self.model_input_generator = ModelInputGenerator(True, noise_scheduler)
 
        self.evaluation2D = Evaluation2D(
            config,  
            self.d2_eval_dataloader, 
            self.train_dataloader,
            None if not self.accelerator.is_main_process else self.tb_summary, 
            self.accelerator,
            self.model_input_generator)
        self.evaluation3D = Evaluation3D(
            config,  
            self.d3_eval_dataloader, 
            None if not self.accelerator.is_main_process else self.tb_summary, 
            self.accelerator)

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
            pipeline = self.pipeline_factory(self.accelerator.unwrap_model(self.model), self.noise_scheduler) 
        pipeline = self.accelerator.prepare(pipeline)
        pipeline.to(self.accelerator.device)
        
        # Evaluate 2D images
        if not self.config.deactivate2Devaluation and ((self.epoch) % self.config.evaluate_epochs == 0 or self.epoch == self.config.num_epochs - 1): 
            self.evaluation2D.evaluate(
                pipeline, 
                self.global_step,  
                parameters = {},
                deactivate_save_model=deactivate_save_model)

        # Evaluate 3D images composed of 2D slices
        if (not self.config.deactivate3Devaluation and ((self.epoch) % self.config.evaluate_3D_epochs == 0 or self.epoch == self.config.num_epochs - 1)): 
            self.evaluation3D.evaluate(pipeline, self.global_step, 
                parameters = {})



 