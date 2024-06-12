
from abc import ABC, abstractmethod 
import os

from accelerate import Accelerator
from diffusers import DiffusionPipeline
from torch.utils.tensorboard import SummaryWriter
import torch  
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

from custom_modules import EvaluationUtils, Training 

class Evaluation2D(ABC):
    """
    Abstract class for evaluating the performance of a 2D diffusion pipeline with 2D images.

    Args:
        config (object): Configuration object containing evaluation settings.
        eval_dataloader (DataLoader): DataLoader for evaluation dataset.
        train_dataloader (DataLoader): DataLoader for training dataset.
        tb_summary (SummaryWriter): SummaryWriter for logging metrics.
        accelerator (Accelerator): Accelerator object for distributed training.
    """

    def __init__(self, config, eval_dataloader: DataLoader, train_dataloader: DataLoader, 
                 tb_summary: SummaryWriter, accelerator: Accelerator):
        self.config = config 
        self.eval_dataloader = eval_dataloader 
        self.train_dataloader = train_dataloader
        self.tb_summary = tb_summary
        self.accelerator = accelerator 
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(self.accelerator.device)
        self.best_ssim = float("inf")

    def _calc_lpip(self, images_1: torch.Tensor, images_2: torch.Tensor) -> torch.Tensor:
        """
        Calculate the LPIPS score between two sets of images and calculate the average.

        Args:
            images_1 (torch.Tensor): Set of images.
            images_2 (torch.Tensor): Set of images.

        Returns:
            torch.Tensor: Float scalar tensor LPIPS score.
        """ 
        lpips = 0
        for i in range(images_1.shape[0]):
            lpips += self.lpips_metric(images_1[i].unsqueeze(0).expand(-1, 3, -1, -1), 
                                       images_2[i].unsqueeze(0).expand(-1, 3, -1, -1))
        lpips /= images_1.shape[0]

        return lpips

    @abstractmethod
    def _start_pipeline(self, pipeline: DiffusionPipeline, batch: torch.Tensor, generator: torch.Generator, 
                        parameters: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Abstract method for starting the diffusion pipeline.

        Args:
            pipeline (DiffusionPipeline): Diffusion pipeline object.
            batch (torch.Tensor): Input batch of data.
            generator (torch.Generator): Random number generator.
            parameters (dict): Additional parameters for the pipeline.

        Returns:
            tuple: Tuple containing the processed images, original images, and masks.
        """
        pass

    def evaluate(self, pipeline: DiffusionPipeline, global_step: int, _get_training_input: Training._get_training_input, 
                 parameters: dict = {}, deactivate_save_model: bool = False) -> None:
        """
        Evaluate the diffusion pipeline and calculate metrics.

        Args:
            pipeline (DiffusionPipeline): Diffusion pipeline object.
            global_step (int): Global step for logging.
            _get_training_input (function): Function for getting training input.
            parameters (dict, optional): Additional parameters for the pipeline. Defaults to {}.
            deactivate_save_model (bool, optional): Whether to deactivate saving the model. Defaults to False.
        """
        # initialize metrics
        metrics = dict()
        metric_list = ["ssim_full", "ssim_out", "ssim_in", "mse_full", "mse_out", "mse_in", 
                       "psnr_full", "psnr_out", "psnr_in", "val_loss", "lpips"] 
        for t in self.config.eval_loss_timesteps:
            metric_list.append(f"val_loss_{t}")
            metric_list.append(f"train_loss_{t}")
        for metric in metric_list:
            metrics[metric] = 0 
        eval_generator = torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed)
            
        # Measure training loss for specific timesteps
        max_iter = len(self.eval_dataloader) if self.config.evaluate_num_batches == -1 else self.config.evaluate_num_batches
        for n_iter, batch_train in enumerate(self.train_dataloader):
            if n_iter >= max_iter:
                break    
            timesteps = torch.tensor(self.config.eval_loss_timesteps, dtype=torch.int, device=self.accelerator.device)
            input, noise, timesteps = _get_training_input(batch_train, generator=eval_generator, timesteps=timesteps)
            noise_pred = pipeline.unet(input, timesteps, return_dict=False)[0]
            for i, t in enumerate(timesteps):
                loss = F.mse_loss(noise_pred[i], noise[i])
                all_loss = self.accelerator.gather_for_metrics(loss).mean() 
                metrics[f"train_loss_{t}"] += all_loss
            del input, noise, timesteps, noise_pred, loss, all_loss

         
        self.progress_bar = tqdm(
            total=len(self.eval_dataloader) if self.config.evaluate_num_batches == -1 else self.config.evaluate_num_batches, 
            disable=not self.accelerator.is_local_main_process) 
        self.progress_bar.set_description(f"Evaluation 2D")

        # Seed dataloader for reproducibility.
        # Normal seeding is not possible because of accelerate.prepare() during training initialization.
        if hasattr(self.eval_dataloader._index_sampler, "sampler"):
            self.eval_dataloader._index_sampler.sampler.generator.manual_seed(self.config.seed)
        else:
            self.eval_dataloader._index_sampler.batch_sampler.sampler.generator.manual_seed(self.config.seed)
        
        for n_iter, batch in enumerate(self.eval_dataloader):
            # Measure validation loss  
            input, noise, timesteps = _get_training_input(batch, generator=eval_generator)
            noise_pred = pipeline.unet(input, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            all_loss = self.accelerator.gather_for_metrics(loss).mean() 
            metrics["val_loss"] += all_loss
            del input, noise, timesteps, noise_pred, loss, all_loss

            # Measure validation loss for specific timesteps
            timesteps = torch.tensor(self.config.eval_loss_timesteps, dtype=torch.int, device=self.accelerator.device)
            input, noise, timesteps = _get_training_input(batch, generator=eval_generator, timesteps=timesteps)
            noise_pred = pipeline.unet(input, timesteps, return_dict=False)[0]
            for i, t in enumerate(timesteps):
                loss = F.mse_loss(noise_pred[i], noise[i])
                all_loss = self.accelerator.gather_for_metrics(loss).mean() 
                metrics[f"val_loss_{t}"] += all_loss 
            del input, noise, timesteps, noise_pred, loss, all_loss 
            torch.cuda.empty_cache() 

            # Run pipeline. The returned masks can be either existing lesions or the synthetic ones
            images, clean_images, masks = self._start_pipeline(
                pipeline,  
                batch,
                eval_generator,
                parameters
            )

            # calculate metrics
            all_clean_images = self.accelerator.gather_for_metrics(clean_images)
            all_images = self.accelerator.gather_for_metrics(images)
            all_masks = self.accelerator.gather_for_metrics(masks)
            metrics["lpips"] += self._calc_lpip(all_clean_images, all_images)
            new_metrics = EvaluationUtils.calc_metrics(all_clean_images, all_images, all_masks)
            for key, value in new_metrics.items(): 
                metrics[key] += value

            self.progress_bar.update(1)
            
            if (self.config.evaluate_num_batches != -1) and (n_iter >= self.config.evaluate_num_batches - 1):
                break 
        
        # calculate average metrics
        for key, value in metrics.items():
            if self.config.evaluate_num_batches == -1:
                metrics[key] /= len(self.eval_dataloader)
            else:
                metrics[key] /= self.config.evaluate_num_batches

        if self.accelerator.is_main_process:
            # log metrics
            EvaluationUtils.log_metrics(self.tb_summary, global_step, metrics, self.config)

            # save last batch as sample images
            list, title_list = self._get_image_lists(images, clean_images, masks, batch)
            image_list = [[to_pil_image(x, mode="L") for x in images] for images in list]
            EvaluationUtils.save_image(image_list, title_list, os.path.join(self.config.output_dir, "samples_2D"), 
                                       global_step, self.config.unet_img_shape)

            # save model
            if not deactivate_save_model and (self.best_ssim > metrics["ssim_in"]):
                self.best_ssim = metrics["ssim_in"]
                pipeline.save_pretrained(self.config.output_dir)
                print("model saved")