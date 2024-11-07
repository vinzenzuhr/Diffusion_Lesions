from abc import ABC, abstractmethod
import os
from typing import Union, Callable

from accelerate import Accelerator 
import diffusers 
from diffusers.training_utils import compute_snr
from diffusers import DDIMScheduler, DDPMScheduler, DiffusionPipeline
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler 
from tqdm.auto import tqdm

from custom_modules import ModelInputGenerator, Evaluation2D, Evaluation3D, Logger
from . import pseudo3D

class Training(ABC):
    """
    The Training class represents the training process for a machine learning model.

    The training loop is implemented in the train method. Accelerator is used for distributed training. 

    Args:
        accelerator (Accelerator): The accelerator used for distributed training.
        model (Union[diffusers.UNet2DModel, pseudo3D.UNet2DModel]): The machine learning model.
        noise_scheduler (Union[DDIMScheduler, DDPMScheduler]): The noise scheduler used for adding 
            noise to the images during training.
        optimizer (Optimizer): The optimizer used for updating the model parameters.
        lr_scheduler (LRScheduler): The learning rate scheduler used for adjusting the learning rate 
            during training.
        train_dataloader (DataLoader): The training dataloader.
        d2_eval_dataloader (DataLoader): The dataloader with 2D evaluation data.
        d3_eval_dataloader (DataLoader): The dataloader with 3D evaluation data.
        model_input_generator (ModelInputGenerator): The model input generator.
        evaluation2D (Evaluation2D): The 2D evaluation module.
        evaluation3D (Evaluation3D): The 3D evaluation module.
        logger (Logger): The logger for logging training metrics.
        pipeline_factory (Callable[[Union[diffusers.UNet2DModel, pseudo3D.UNet2DModel], Union[DDIMScheduler, DDPMScheduler]], DiffusionPipeline]): 
            The factory function for creating the diffusion pipeline.
        num_epochs (int): The number of training epochs.
        evaluate_2D_epochs (Union[float, int]): The interval at which to evaluate the model on 2D images.
            With [0,1] the model can be evaluated during an epoch.
        evaluate_3D_epochs (int): The interval at which to evaluate the model on 3D images.
        min_snr_loss (bool, optional): A flag indicating whether to use the minimum SNR loss. Defaults to False.
        snr_gamma (int, optional): The gamma value for the SNR loss. Defaults to 5.
        evaluate_unconditional_img (bool, optional): A flag indicating whether to evaluate and save 
            unconditional images. Defaults to False.
        deactivate_2D_evaluation (bool, optional): A flag indicating whether to deactivate 
            2D evaluation. Defaults to False.
        deactivate_3D_evaluation (bool, optional): A flag indicating whether to deactivate 
            3D evaluation. Defaults to True.
        evaluation_pipeline_parameters (dict, optional): Additional parameters for the evaluation 
            pipeline. Defaults to {}.
        debug (bool, optional): A flag indicating whether to run in debug mode, which stops the 
            training loop after one epoch. Defaults to False.
    """

    def __init__(
            self,  
            accelerator: Accelerator, 
            model: Union[diffusers.UNet2DModel, pseudo3D.UNet2DModel], 
            noise_scheduler: Union[DDIMScheduler, DDPMScheduler],
            optimizer: Optimizer, 
            lr_scheduler: LRScheduler, 
            train_dataloader: DataLoader, 
            d2_eval_dataloader: DataLoader, 
            d3_eval_dataloader: DataLoader, 
            model_input_generator: ModelInputGenerator,
            evaluation2D: Evaluation2D,
            evaluation3D: Evaluation3D,
            logger: Logger,
            pipeline_factory: Callable[[Union[diffusers.UNet2DModel, pseudo3D.UNet2DModel], Union[DDIMScheduler, DDPMScheduler]], DiffusionPipeline], 
            num_epochs: int,
            evaluate_2D_epochs: Union[float, int],
            evaluate_3D_epochs: int,
            min_snr_loss: bool = False,
            snr_gamma = 5,
            evaluate_unconditional_img: bool = False,
            deactivate_2D_evaluation: bool = False, 
            deactivate_3D_evaluation: bool = True, 
            evaluation_pipeline_parameters: dict = {},
            debug: bool = False,
            ):

        self.accelerator = accelerator
        self.logger = logger 
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler  
        self.train_dataloader = train_dataloader
        self.d2_eval_dataloader = d2_eval_dataloader
        self.d3_eval_dataloader = d3_eval_dataloader
        self.model_input_generator = model_input_generator
        self.evaluation2D = evaluation2D
        self.evaluation3D = evaluation3D
        self.pipeline_factory = pipeline_factory
        self.num_epochs = num_epochs
        self.evaluate_2D_epochs = evaluate_2D_epochs
        self.evaluate_3D_epochs = evaluate_3D_epochs 
        self.min_snr_loss = min_snr_loss    
        self.evaluate_unconditional_img = evaluate_unconditional_img
        self.deactivate_2D_evaluation = deactivate_2D_evaluation 
        self.deactivate_3D_evaluation = deactivate_3D_evaluation
        self.evaluation_pipeline_parameters = evaluation_pipeline_parameters
        self.debug = debug

        self.epoch = 0
        self.global_step = 0 

        if self.min_snr_loss:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            timesteps=torch.arange(1000, device=self.accelerator.device)
            snr=compute_snr(self.noise_scheduler, timesteps) 
            self.loss_weights=(
                torch.stack(
                    [snr, snr_gamma * torch.ones_like(timesteps, device=self.accelerator.device)], dim=1
                    ).min(dim=1)[0] / snr
            )
            # For zero-terminal SNR, we have to handle the case where a sigma of Zero results in a Inf value.
            self.loss_weights[snr==0] = 1.0

    def evaluate(self, pipeline: DiffusionPipeline = None, deactivate_save_model: bool = False): 
        """ 
        Evaluates the model on the evaluation datasets. 
        
        Args:
            pipeline (DiffusionPipeline): The diffusion pipeline.
            deactivate_save_model (bool): A flag indicating whether to deactivate saving the model 
                during evaluation.
        """
        self.model.eval()
        # Create pipeline if not given
        if pipeline is None:
            pipeline = self.pipeline_factory(self.accelerator.unwrap_model(self.model), self.noise_scheduler) 
        pipeline = self.accelerator.prepare(pipeline)
        pipeline.to(self.accelerator.device) 
        
        # Evaluate 2D images
        if self.epoch % self.evaluate_2D_epochs == 0 or self.epoch == self.num_epochs - 1 or self.evaluate_2D_epochs < 1: 
            if not self.deactivate_2D_evaluation:
                self.evaluation2D.evaluate(
                    pipeline, 
                    self.global_step,  
                    parameters = self.evaluation_pipeline_parameters,
                    deactivate_save_model=deactivate_save_model)
            if self.evaluate_unconditional_img: 
                self.evaluation2D.save_unconditional_img(self.accelerator.unwrap_model(self.model), 
                                                         self.noise_scheduler, self.global_step) 
        
        # Evaluate 3D images composed of 2D slices
        if (
            not self.deactivate_3D_evaluation 
            and (
                self.epoch % self.evaluate_3D_epochs == 0 
                or self.epoch == self.num_epochs - 1
                or self.evaluate_3D_epochs < 1
                )
            ): 
            self.evaluation3D.evaluate(
                pipeline, 
                self.global_step, 
                parameters = self.evaluation_pipeline_parameters,
                )
    
    def train(self):
        """ Trains the model using the training dataset. """
        for self.epoch in torch.arange(self.num_epochs):
            self.progress_bar = tqdm(total=len(self.train_dataloader), disable=not self.accelerator.is_local_main_process) 
            self.progress_bar.set_description(f"Epoch {self.epoch}") 
            self.model.train() 
            for idx, batch in enumerate(self.train_dataloader):

                input, noise, timesteps = self.model_input_generator.get_model_input(batch)
                 
                with self.accelerator.accumulate(self.model):
                    # Predict the noise residual
                    noise_pred = self.model(input, timesteps, return_dict=False)[0]

                    if(self.min_snr_loss):
                        # calculate the mse loss, mean over the non-batch dimensions, rebalance 
                        # sample-wise losses with the timestep dependet loss weights and take the mean
                        mse_weights = self.loss_weights[timesteps]
                        loss = F.mse_loss(noise_pred, noise, reduction="none") 
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_weights 
                        loss = loss.mean()
                    else:
                        loss = F.mse_loss(noise_pred, noise)

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        total_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0).cpu().detach().item()
                    else:
                        total_norm = None
        
                    # do learning step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad() 

                self.progress_bar.update(1)

                if self.accelerator.is_main_process: 
                    tags = ["weighted_train_loss", "lr", "total_norm"]
                    values = [loss.cpu().detach().item(), self.lr_scheduler.get_last_lr()[0], total_norm]
                    logs = zip(tags, values)
                    self.logger.log_train_metrics(self.global_step, logs)

                    logs = dict(logs)
                    logs["step"] = self.global_step
                    self.progress_bar.set_postfix(**logs) 


                self.global_step += 1 

                if idx % int(self.evaluate_2D_epochs * len(self.train_dataloader)) == 0:
                    self.evaluate()

                if self.debug:
                    break
            
            self.evaluate()