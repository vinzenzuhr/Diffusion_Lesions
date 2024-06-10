from custom_modules import get_dataloader

from abc import ABC, abstractmethod
from accelerate import Accelerator
import numpy as np
import os
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler 
from diffusers.training_utils import compute_snr
import torch.nn.functional as F
from tqdm.auto import tqdm

#from accelerate.utils import InitProcessGroupKwargs
#from datetime import timedelta


class Training(ABC):
    def __init__(
            self, 
            config, 
            model, 
            noise_scheduler, 
            optimizer, 
            lr_scheduler, 
            datasetTrain, 
            datasetEvaluation, 
            dataset3DEvaluation, 
            evaluation2D, 
            evaluation3D, 
            pipelineFactory, 
            multi_sample=False,
            min_snr_loss=False,
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
            project_dir=os.path.join(config.output_dir, "tensorboard"), #evt. delete
            #kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=2 * 1800))],
        ) 
        
        self.train_dataloader = get_dataloader(dataset = datasetTrain, batch_size = config.train_batch_size, num_workers=self.config.num_dataloader_workers ,random_sampler=True, seed=self.config.seed, multi_sample=multi_sample)
        self.d2_eval_dataloader = get_dataloader(dataset = datasetEvaluation, batch_size = config.eval_batch_size, num_workers=self.config.num_dataloader_workers, random_sampler=False, seed=self.config.seed, multi_sample=multi_sample)
        self.d3_eval_dataloader = get_dataloader(dataset = dataset3DEvaluation, batch_size = 1, num_workers=self.config.num_dataloader_workers, random_sampler=False, seed=self.config.seed, multi_sample=False)        

        if self.accelerator.is_main_process:
            #setup tensorboard
            self.tb_summary = SummaryWriter(config.output_dir, purge_step=0)
            self.log_meta_logs()
            
            if config.output_dir is not None:
                os.makedirs(config.output_dir, exist_ok=True) 
            self.accelerator.init_trackers("train_example") #evt. delete

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

        os.makedirs(config.output_dir, exist_ok=True)  #evt. delete

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
        #log at tensorboard
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

        for scalar in scalars:
            if hasattr(self.config, scalar) and getattr(self.config, scalar) is not None:
                self.tb_summary.add_scalar(scalar, getattr(self.config, scalar), 0)
        for text in texts:
            if hasattr(self.config, text) and getattr(self.config, text) is not None:
                self.tb_summary.add_text(text, getattr(self.config, text), 0)
        
        self.tb_summary.add_scalar("len(train_dataloader)", len(self.train_dataloader), 0)
        self.tb_summary.add_scalar("len(d2_eval_dataloader)", len(self.d2_eval_dataloader), 0)
        self.tb_summary.add_scalar("len(d3_eval_dataloader)", len(self.d3_eval_dataloader), 0) 
        if self.config.t1n_target_shape:
            self.tb_summary.add_scalar("t1n_target_shape_x", self.config.t1n_target_shape[0], 0) 
            self.tb_summary.add_scalar("t1n_target_shape_y", self.config.t1n_target_shape[1], 0) 

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

        

    
    def _save_logs(self, loss, total_norm):
        
        logs = {"weighted_train_loss": loss.cpu().detach().item(), "lr": self.lr_scheduler.get_last_lr()[0], "step": self.global_step}
        self.tb_summary.add_scalar("weighted_train_loss", logs["weighted_train_loss"], self.global_step)
        self.tb_summary.add_scalar("lr", logs["lr"], self.global_step)
        if total_norm:
            self.tb_summary.add_scalar("total_norm", total_norm, self.global_step) 
    
        self.progress_bar.set_postfix(**logs)
        #accelerator.log(logs, step=global_step)

    def _get_noisy_images(self, clean_images, generator=None, timesteps=None):

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
    def _get_training_input(self, batch, generator=None, timesteps=None):
        pass

    @abstractmethod
    def evaluate(self):
        pass
    
    def train(self):
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
                        mse_weights = self.loss_weights[timesteps]
                        # mean over the non-batch dimensions, rebalance sample-wise losses with their respective loss weights and then take the mean
                        loss = F.mse_loss(noise_pred, noise, reduction="none") 
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_weights 
                        loss = loss.mean()
                    else:
                        loss = F.mse_loss(noise_pred, noise)

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        total_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    else:
                        total_norm = None
        
                    #log gradient norm 
                    #parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
                    #if len(parameters) == 0:
                    #    total_norm = 0.0
                    #else: 
                    #    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).cpu() for p in parameters]), 2.0).item()

                    #do learning step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad() 

                self.progress_bar.update(1)

                # save logs
                if self.accelerator.is_main_process:
                    self._save_logs(loss, total_norm)

                self.global_step += 1 

                if self.config.debug:
                    break
            
            self.evaluate()