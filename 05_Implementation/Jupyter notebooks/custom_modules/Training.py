from abc import ABC, abstractmethod
from accelerate import Accelerator
import numpy as np
import os
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler 
import torch.nn.functional as F
from tqdm.auto import tqdm

class Training(ABC):
    def __init__(self, config, model, noise_scheduler, optimizer, lr_scheduler, datasetTrain, datasetEvaluation, dataset3DEvaluation, evaluation2D, evaluation3D, pipelineFactory):
        self.config = config
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler  
        self.evaluation2D = evaluation2D
        self.evaluation3D = evaluation3D
        self.pipelineFactory = pipelineFactory

        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps, 
            project_dir=os.path.join(config.output_dir, "tensorboard"), #evt. delete
        )

        if self.accelerator.is_main_process:
            #setup tensorboard
            self.tb_summary = SummaryWriter(config.output_dir, purge_step=0)
            self._meta_logs()
            
            if config.output_dir is not None:
                os.makedirs(config.output_dir, exist_ok=True) 
            self.accelerator.init_trackers("train_example") #evt. delete

        self.train_dataloader = self._get_dataloader(dataset = datasetTrain, batch_size = config.train_batch_size, random_sampler=True)
        self.d2_eval_dataloader = self._get_dataloader(dataset = datasetEvaluation, batch_size = config.eval_batch_size, random_sampler=False)
        self.d3_eval_dataloader = self._get_dataloader(dataset = dataset3DEvaluation, batch_size = 1, random_sampler=False) 

        self.model, self.optimizer, self.train_dataloader, self.d2_eval_dataloader, self.d3_eval_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.d2_eval_dataloader, self.d3_eval_dataloader, self.lr_scheduler
        )

        os.makedirs(config.output_dir, exist_ok=True)  #evt. delete

        self.epoch = 0
        self.global_step = 0

    def _get_dataloader(self, dataset, batch_size, num_workers=4, random_sampler=False): 
        sampler = RandomSampler(dataset, generator=(None if random_sampler else torch.cuda.manual_seed_all(self.config.seed)))
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, worker_init_fn=(None if random_sampler else self._reset_seed))
    
    def _get_random_masks(self, n, generator=None):
        #create circular mask with random center around the center point of the pictures and a radius between 3 and 50 pixels
        center=torch.normal(mean=self.config.image_size/2, std=30, size=(n,2), generator=generator) # 30 is chosen by inspection
        low=3
        high=50
        radius=torch.rand(n, generator=generator)*(high-low)+low # get radius between 3 and 50 from uniform distribution 

        #Test case
        #center=torch.tensor([[0,255],[0,255]]) 
        #radius=torch.tensor([2,2])
        
        Y, X = [torch.arange(self.config.image_size)[:,None],torch.arange(self.config.image_size)[None,:]] # gives two vectors, each containing the pixel locations. There's a column vector for the column indices and a row vector for the row indices.
        dist_from_center = torch.sqrt((X.T - center[:,0])[None,:,:]**2 + (Y-center[:,1])[:,None,:]**2) # creates matrix with euclidean distance to center
        dist_from_center = dist_from_center.permute(2,0,1) 

        #Test case
        #print(dist_from_center[0,0,0]) #=255
        #print(dist_from_center[0,0,255]) #=360.624
        #print(dist_from_center[0,255,0]) #=0
        #print(dist_from_center[0,255,255]) #=255
        #print(dist_from_center[0,127,127]) #=180.313 
        
        masks = dist_from_center < radius[:,None,None] # creates mask for pixels which are inside the radius. 
        masks = masks[:,None,:,:].int() 
        return masks

    def _meta_logs(self):
        #log at tensorboard
        self.tb_summary.add_scalar("image_size", self.config.image_size, 0)
        self.tb_summary.add_scalar("train_batch_size", self.config.train_batch_size, 0)
        self.tb_summary.add_scalar("eval_batch_size", self.config.eval_batch_size, 0)
        self.tb_summary.add_scalar("num_epochs", self.config.num_epochs, 0)
        self.tb_summary.add_scalar("learning_rate", self.config.learning_rate, 0)
        self.tb_summary.add_scalar("lr_warmup_steps", self.config.lr_warmup_steps, 0)
        self.tb_summary.add_scalar("evaluate_epochs", self.config.evaluate_epochs, 0)
        self.tb_summary.add_scalar("evaluate_3D_epochs", self.config.evaluate_3D_epochs, 0) 
        self.tb_summary.add_text("mixed_precision", self.config.mixed_precision, 0) 
        self.tb_summary.add_scalar("train_only_connected_masks", self.config.train_only_connected_masks, 0)
        self.tb_summary.add_scalar("eval_only_connected_masks", self.config.eval_only_connected_masks, 0) 
        self.tb_summary.add_scalar("debug", self.config.debug, 0) 
        self.tb_summary.add_text("mode", self.config.mode, 0) 
        self.tb_summary.add_text("model", self.config.model, 0) #config.model = "UNet2DModel"
        self.tb_summary.add_text("noise_scheduler", self.config.noise_scheduler, 0) #config.noise_scheduler = "DDIMScheduler(num_train_timesteps=1000)"
        self.tb_summary.add_text("lr_scheduler", self.config.lr_scheduler, 0) #config.lr_scheduler = "cosine_schedule_with_warmup"
        self.tb_summary.add_text("conditional_data", self.config.conditional_data, 0) #config.conditional_data = "Lesions"

    def _reset_seed(self, worker_id=0): 
        np.random.seed(self.config.seed) 
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        random.seed(self.config.seed)
        return
    
    def _save_logs(self, loss, total_norm):
        
        logs = {"loss": loss.cpu().detach().item(), "lr": self.lr_scheduler.get_last_lr()[0], "total_norm": total_norm, "step": self.global_step}
        self.tb_summary.add_scalar("loss", logs["loss"], self.global_step)
        self.tb_summary.add_scalar("lr", logs["lr"], self.global_step)
        self.tb_summary.add_scalar("total_norm", logs["total_norm"], self.global_step) 
    
        self.progress_bar.set_postfix(**logs)
        #accelerator.log(logs, step=global_step)

    def _get_noisy_images(self, clean_images, generator=None):

        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape, device=clean_images.device, generator=generator)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
            dtype=torch.int64, generator=generator
        ) 

        # Add noise to the voided images according to the noise magnitude at each timestep (forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps) 
        
        return noisy_images, noise, timesteps

    @abstractmethod
    def _get_training_input(self, batch, generator=None):
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
                    loss = F.mse_loss(noise_pred, noise)
                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
        
                    #log gradient norm 
                    parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
                    if len(parameters) == 0:
                        total_norm = 0.0
                    else: 
                        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).cpu() for p in parameters]), 2.0).item()

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