from abc import ABC, abstractmethod
import os
from typing import Union, Callable

from accelerate import Accelerator 
import diffusers 
from diffusers.training_utils import compute_snr
from diffusers import DDIMScheduler, DDPMScheduler, DiffusionPipeline
import torch
from torch.utils.tensorboard import SummaryWriter 
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler 
from tqdm.auto import tqdm

from custom_modules import get_dataloader, DatasetMRI2D, DatasetMRI3D, Evaluation2D, Evaluation3D
import pseudo3D

class Training(ABC):
    """
    Abstract base class for training a model.

    The training loop is implemented in the train method. Accelerator is used for distributed training. 

    Args:
        config (object): The configuration object containing various training parameters.
        model (Union[diffusers.UNet2DModel, pseudo3D.UNet2DModel]): The model to be trained.
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
        self.config = config
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler  
        self.pipelineFactory = pipelineFactory
        self.min_snr_loss=min_snr_loss
 
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,  
        ) 
        
        self.train_dataloader = get_dataloader(dataset=datasetTrain, batch_size = config.train_batch_size, 
                                               num_workers=self.config.num_dataloader_workers, random_sampler=True, 
                                               seed=self.config.seed, multi_slice=sorted_slice_sample_size > 1)
        self.d2_eval_dataloader = get_dataloader(dataset=datasetEvaluation, batch_size = config.eval_batch_size, 
                                                 num_workers=self.config.num_dataloader_workers, random_sampler=False, 
                                                 seed=self.config.seed, multi_slice=sorted_slice_sample_size > 1)
        self.d3_eval_dataloader = get_dataloader(dataset=dataset3DEvaluation, batch_size = 1, 
                                                 num_workers=self.config.num_dataloader_workers, random_sampler=False, 
                                                 seed=self.config.seed, multi_slice=False) 

        if self.accelerator.is_main_process:
            #setup tensorboard
            self.tb_summary = SummaryWriter(config.output_dir, purge_step=0)
            self.log_meta_logs()
            
            if config.output_dir is not None:
                os.makedirs(config.output_dir, exist_ok=True) 
            self.accelerator.init_trackers("train_example") #maybe delete

        self.model, self.optimizer, self.train_dataloader, self.d2_eval_dataloader, self.d3_eval_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.d2_eval_dataloader, self.d3_eval_dataloader, self.lr_scheduler
        )
 
        self.evaluation2D = evaluation2D(
            config,  
            self.d2_eval_dataloader, 
            self.train_dataloader,
            None if not self.accelerator.is_main_process else self.tb_summary, 
            self.accelerator)
        self.evaluation3D = evaluation3D(
            config,  
            self.d3_eval_dataloader, 
            None if not self.accelerator.is_main_process else self.tb_summary, 
            self.accelerator)

        os.makedirs(config.output_dir, exist_ok=True)  #maybe delete

        self.epoch = 0
        self.global_step = 0 

        if self.min_snr_loss:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            timesteps=torch.arange(1000, device=self.accelerator.device)
            snr=compute_snr(self.noise_scheduler, timesteps)
            gamma=self.config.snr_gamma
            self.loss_weights=(
                torch.stack([snr, gamma * torch.ones_like(timesteps, device=self.accelerator.device)], dim=1).min(dim=1)[0] / snr
            )
            # For zero-terminal SNR, we have to handle the case where a sigma of Zero results in a Inf value.
            self.loss_weights[snr==0] = 1.0

    def log_meta_logs(self):
        """ Logs metadata information from the config to TensorBoard and CSV file. """
        scalars = [              
            "train_batch_size",
            "eval_batch_size",
            "num_epochs",
            "learning_rate",
            "lr_warmup_steps",
            "evaluate_epochs",
            "evaluate_num_batches",
            "evaluate_3D_epochs",
            "train_only_connected_masks",
            "eval_only_connected_masks",
            "debug",
            "brightness_augmentation",
            "intermediate_timestep",
            "jump_n_sample",
            "jump_length",
            "gradient_accumulation_steps",
            "proportionTrainingCircularMasks",
            "use_min_snr_loss",
            "snr_gamma",
        ] 
        texts = [
            "mixed_precision",
            "mode",
            "model",
            "noise_scheduler",
            "lr_scheduler", 
            "add_lesion_technique",
            "dataset_train_path",
            "dataset_eval_path",
            "restrict_train_slices",
            "restrict_eval_slices",
        ] 

        #log at tensorboard
        for scalar in scalars:
            if hasattr(self.config, scalar) and getattr(self.config, scalar) is not None:
                self.tb_summary.add_scalar(scalar, getattr(self.config, scalar), 0)
        for text in texts:
            if hasattr(self.config, text) and getattr(self.config, text) is not None:
                self.tb_summary.add_text(text, getattr(self.config, text), 0)
        self.tb_summary.add_scalar("len(train_dataloader)", len(self.train_dataloader), 0)
        self.tb_summary.add_scalar("len(d2_eval_dataloader)", len(self.d2_eval_dataloader), 0)
        self.tb_summary.add_scalar("len(d3_eval_dataloader)", len(self.d3_eval_dataloader), 0) 
        if self.config.target_shape:
            self.tb_summary.add_scalar("target_shape_x", self.config.target_shape[0], 0) 
            self.tb_summary.add_scalar("target_shape_y", self.config.target_shape[1], 0) 

        #log to csv
        if self.config.log_csv:
            with open(os.path.join(self.config.output_dir, "metrics.csv"), "w") as f:
                for scalar in scalars:
                    if hasattr(self.config, scalar) and getattr(self.config, scalar) is not None:
                        f.write(f"{scalar}:{getattr(self.config, scalar)},")
                for text in texts:
                    if hasattr(self.config, text) and getattr(self.config, text) is not None:
                        f.write(f"{text}:{getattr(self.config, text)},")
                f.write(f"len(train_dataloader):{len(self.train_dataloader)},")
                f.write(f"len(d2_eval_dataloader):{len(self.d2_eval_dataloader)},")
                f.write(f"len(d3_eval_dataloader):{len(self.d3_eval_dataloader)}")
                f.write("\n")

    def _save_logs(self, loss: int, total_norm: int):
        """
        Saves the training logs to the TensorBoard summary and updates the progress bar.

        Args:
            loss (int): The weighted training loss.
            total_norm (int): The total norm of the gradients.
        """
        logs = {
            "weighted_train_loss": loss, 
            "lr": self.lr_scheduler.get_last_lr()[0], 
            "step": self.global_step}
        self.tb_summary.add_scalar("weighted_train_loss", logs["weighted_train_loss"], self.global_step)
        self.tb_summary.add_scalar("lr", logs["lr"], self.global_step)
        if total_norm:
            self.tb_summary.add_scalar("total_norm", total_norm, self.global_step) 

        self.progress_bar.set_postfix(**logs)

    def _get_noisy_images(self, clean_images: torch.tensor, generator: torch.Generator = None, 
                          timesteps: torch.tensor = None) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Generates noisy images by adding random gaussian noise to the clean images. 

        Args:
            clean_images (torch.tensor): The clean images to add noise to.
            generator (torch.Generator, optional): The random number generator. Defaults to None.
            timesteps (torch.tensor, optional): Predefined timesteps for diffusing each image. 
                Defaults to None.

        Returns:
            tuple[torch.tensor, torch.tensor, torch.tensor]: A tuple containing the noisy images, 
                the used noise, and the timesteps.
        """
        
        # Sample noise to add to the images 
        noise = torch.randn(clean_images.shape, device=clean_images.device, generator=generator)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        if timesteps is None:
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64, generator=generator
            )
        assert timesteps.shape[0] == bs 

        # Add noise to the voided images according to the noise magnitude at each timestep (forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps) 
        
        return noisy_images, noise, timesteps

    @abstractmethod
    def _get_training_input(self, batch: torch.tensor, generator: torch.Generator = None, timesteps: torch.tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the training input for the model.

        Args:
            batch (torch.tensor): The input batch of data.
            generator (torch.Generator, optional): The random number generator. Defaults to None.
            timesteps (torch.tensor, optional): Predefined timesteps for diffusing each image. 
                Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the input for the model, 
                the sampled noise, and the used timesteps.
        """
        pass

    @abstractmethod
    def evaluate(self, pipeline: DiffusionPipeline = None, deactivate_save_model: bool = False):
        """ 
        Evaluates the model on the evaluation datasets. 
        
        Args:
            pipeline: The diffusion pipeline.
            deactivate_save_model: A flag indicating whether to deactivate saving the model during evaluation.
        """

        pass
    
    def train(self):
        """ Trains the model using the training dataset. """
        for self.epoch in torch.arange(self.config.num_epochs):
            self.progress_bar = tqdm(total=len(self.train_dataloader), disable=not self.accelerator.is_local_main_process) 
            self.progress_bar.set_description(f"Epoch {self.epoch}") 
            self.model.train() 
            for batch in self.train_dataloader:

                input, noise, timesteps = self._get_training_input(batch)
                 
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
                    self._save_logs(loss.cpu().detach().item(), total_norm)

                self.global_step += 1 

                if self.config.debug:
                    break
            
            self.evaluate()