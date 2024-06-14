
from abc import ABC, abstractmethod 
import os
from pathlib import Path
from typing import Union


from accelerate import Accelerator
import diffusers
from diffusers import DiffusionPipeline, DDIMPipeline, DDIMScheduler, DDPMScheduler, DiffusionPipeline
from diffusers.utils import make_image_grid
import PIL 
import torch  
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

from custom_modules import EvaluationUtils, ModelInputGenerator, Logger
from . import pseudo3D

class Evaluation2D(ABC):
    """
    Abstract class for evaluating the performance of a 2D diffusion pipeline with 2D images.

    Args:
        eval_dataloader (DataLoader): DataLoader for evaluation data.
        train_dataloader (DataLoader): DataLoader for training data.
        logger (Logger): Object for logging.
        accelerator (Accelerator): Accelerator object for distributed training.
        num_inference_steps (int): Number of inference steps.
        model_input_generator (ModelInputGenerator): ModelInputGenerator object for generating different model inputs.
        output_dir (str): Output directory for saving results.
        eval_loss_timesteps (list[int]): List of timesteps to evalute validation loss.
        evaluate_num_batches (int, optional): Number of batches to evaluate. Defaults to -1 (all batches).
        seed (int, optional): The random seed for reproducibility. Defaults to None.
    """

    def __init__(self, eval_dataloader: DataLoader, train_dataloader: DataLoader, 
                 logger: Logger, accelerator: Accelerator, num_inference_steps: int, 
                 model_input_generator: ModelInputGenerator, output_dir: str, eval_loss_timesteps: list[int], 
                 evaluate_num_batches: int = -1, seed: int = None):
        self.eval_dataloader = eval_dataloader 
        self.train_dataloader = train_dataloader
        self.logger = logger
        self.accelerator = accelerator
        self.num_inference_steps = num_inference_steps 
        self.best_ssim = float("inf")
        self.model_input_generator = model_input_generator
        self.output_dir = output_dir
        self.eval_loss_timesteps = eval_loss_timesteps
        self.evaluate_num_batches = evaluate_num_batches
        self.seed = seed

        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(self.accelerator.device)

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

    def evaluate(self, pipeline: DiffusionPipeline, global_step: int, 
                 parameters: dict = {}, deactivate_save_model: bool = False):
        """
        Evaluate the diffusion pipeline and calculate metrics.

        Args:
            pipeline (DiffusionPipeline): Diffusion pipeline object.
            global_step (int): Global step for logging. 
            parameters (dict, optional): Additional parameters for the pipeline. Defaults to {}.
            deactivate_save_model (bool, optional): Whether to deactivate saving the model. Defaults to False.
        """
        # initialize metrics
        metrics = dict()
        metric_list = ["ssim_full", "ssim_out", "ssim_in", "mse_full", "mse_out", "mse_in", 
                       "psnr_full", "psnr_out", "psnr_in", "val_loss", "lpips"] 
        for t in self.eval_loss_timesteps:
            metric_list.append(f"val_loss_{t}")
            metric_list.append(f"train_loss_{t}")
        for metric in metric_list:
            metrics[metric] = 0 
        eval_generator = torch.Generator(device=self.accelerator.device).manual_seed(self.seed)
            
        # Measure training loss for specific timesteps
        max_iter = len(self.eval_dataloader) if self.evaluate_num_batches == -1 else self.evaluate_num_batches
        for n_iter, batch_train in enumerate(self.train_dataloader):
            if n_iter >= max_iter:
                break    
            timesteps = torch.tensor(self.eval_loss_timesteps, dtype=torch.int, device=self.accelerator.device)
            input, noise, timesteps = self.model_input_generator.get_model_input(batch_train, generator=eval_generator, timesteps=timesteps)
            noise_pred = pipeline.unet(input, timesteps, return_dict=False)[0]
            for i, t in enumerate(timesteps):
                loss = F.mse_loss(noise_pred[i], noise[i])
                all_loss = self.accelerator.gather_for_metrics(loss).mean() 
                metrics[f"train_loss_{t}"] += all_loss
            del input, noise, timesteps, noise_pred, loss, all_loss

         
        self.progress_bar = tqdm(
            total=len(self.eval_dataloader) if self.evaluate_num_batches == -1 else self.evaluate_num_batches, 
            disable=not self.accelerator.is_local_main_process) 
        self.progress_bar.set_description(f"Evaluation 2D")

        # Seed dataloader for reproducibility.
        # Normal seeding is not possible because of accelerate.prepare() during training initialization.
        if hasattr(self.eval_dataloader._index_sampler, "sampler"):
            self.eval_dataloader._index_sampler.sampler.generator.manual_seed(self.seed)
        else:
            self.eval_dataloader._index_sampler.batch_sampler.sampler.generator.manual_seed(self.seed)
        
        for n_iter, batch in enumerate(self.eval_dataloader):
            # Measure validation loss  
            input, noise, timesteps = self.model_input_generator.get_model_input(batch, generator=eval_generator)
            noise_pred = pipeline.unet(input, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            all_loss = self.accelerator.gather_for_metrics(loss).mean() 
            metrics["val_loss"] += all_loss
            del input, noise, timesteps, noise_pred, loss, all_loss

            # Measure validation loss for specific timesteps
            timesteps = torch.tensor(self.eval_loss_timesteps, dtype=torch.int, device=self.accelerator.device)
            input, noise, timesteps = self.model_input_generator.get_model_input(batch, generator=eval_generator, timesteps=timesteps)
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
            
            if (self.evaluate_num_batches != -1) and (n_iter >= self.evaluate_num_batches - 1):
                break 
        
        # calculate average metrics
        for key, value in metrics.items():
            if self.evaluate_num_batches == -1:
                metrics[key] /= len(self.eval_dataloader)
            else:
                metrics[key] /= self.evaluate_num_batches

        if self.accelerator.is_main_process:
            # log metrics
            self.logger.log_eval_metrics(global_step, metrics, self.output_dir)

            # save last batch as sample images
            list, title_list = self._get_image_lists(images, clean_images, masks, batch)
            image_list = [[to_pil_image(x, mode="L") for x in images] for images in list]
            self.save_image(image_list, title_list, os.path.join(self.output_dir, "samples_2D"), 
                                       global_step)

            # save model
            if not deactivate_save_model and (self.best_ssim < metrics["ssim_in"]):
                self.best_ssim = metrics["ssim_in"]
                pipeline.save_pretrained(self.output_dir)
                print("model saved")

    def save_image(self, image_lists: list[list[PIL.Image]], titles: list[str], path: Path, global_step: int):
        """
        Save a grid of images to the specified path.

        Args:
            image_lists (list[list[PIL.Image]]): A list of lists of PIL.Image objects representing the images to be saved.
            titles (list[str]): A list of titles for each image grid.
            path (Path): The path where the images will be saved.
            global_step (int): The global step used for naming the saved images.

        Raises:
            ValueError: If the number of images in the list is exceeds the limit.

        Returns:
            None
        """
        os.makedirs(path, exist_ok=True)
        for image_list, title in zip(image_lists, titles):             
            if len(image_list) > 16:
                raise ValueError("Number of images in list must be less than 16")
            missing_num = 16 - len(image_list)
            for _ in range(missing_num):
                image_list.append(PIL.Image.new("L", image_list[0].shape, 0))
            image_grid = make_image_grid(image_list, rows=4, cols=4)
            image_grid.save(f"{path}/{title}_{global_step:07d}.png")
        print("image saved")

    def save_unconditional_img(self, model: Union[diffusers.UNet2DModel, pseudo3D.UNet2DModel], 
                               noise_scheduler: Union[DDIMScheduler, DDPMScheduler], global_step: int,):
        """
        Save images generated by the uncondtional model.

        Args:
            model (Union[diffusers.UNet2DModel, pseudo3D.UNet2DModel]): Model for generating images.
            noise_scheduler (Union[DDIMScheduler, DDPMScheduler]): Noise scheduler for the model.
            global_step (int): Global step for logging.
        """
        unconditional_pipeline = DDIMPipeline(
            unet=model, 
            scheduler=noise_scheduler)
        unconditional_pipeline = self.accelerator.prepare(unconditional_pipeline) 
        images = unconditional_pipeline(
            batch_size = self.eval_dataloader.batch_size,
            generator=torch.Generator(device=self.accelerator.device).manual_seed(self.seed), 
            output_type = "pil",
            num_inference_steps = self.num_inference_steps 
        )[0] 
        self.save_image([images],["unconditional_images"], os.path.join(self.output_dir, "samples_2D"),
                        global_step)