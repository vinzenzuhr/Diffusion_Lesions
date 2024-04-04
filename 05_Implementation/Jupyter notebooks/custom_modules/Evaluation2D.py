#setup evaluation
from abc import ABC, abstractmethod
import EvaluationUtils
from diffusers.utils import make_image_grid
import numpy as np
import os
from pathlib import Path
import torch.nn.functional as F
import PIL 
import random
from skimage.metrics import structural_similarity, mean_squared_error
import torch
from torcheval.metrics import PeakSignalNoiseRatio
from torchvision.transforms.functional import to_pil_image

class Evaluation2D(ABC):
    def __init__(self, config, pipeline, dataloader, tb_summary, accelerator, train_env):
        self.config = config
        self.pipeline = pipeline
        self.dataloader = dataloader
        self.tb_summary = tb_summary
        self.accelerator = accelerator
        self.train_env = train_env 

    def _reset_seed(self, seed): 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        return

    def _save_image(self, images: list[list[PIL.Image]], titles: list[str], path: Path, global_step: int):
        os.makedirs(path, exist_ok=True)
        for image_list, title in zip(images, titles): 
            image_grid = make_image_grid(image_list, rows=int(len(image_list)**0.5), cols=int(len(image_list)**0.5))
            image_grid.save(f"{path}/{title}_{global_step:07d}.png")
        print("image saved")

    @abstractmethod
    def _start_pipeline(self, clean_images, masks, parameters):
        pass

    def evaluate(self, global_step, parameters={}):
        #initialize metrics
        metrics = dict()
        metric_list = ["ssim_full", "ssim_out", "ssim_in", "mse_full", "mse_out", "mse_in", "psnr_full", "psnr_out", "psnr_in", "val_loss"] 
        for metric in metric_list:
            metrics[metric] = 0

        self._reset_seed(self.config.seed)
        for n_iter, batch in enumerate(self.dataloader): 
            if n_iter >= self.config.evaluate_num_batches:
                break

            # calc validation loss
            input, noise, timesteps = self.train_env._get_training_input(batch) 
            noise_pred = self.pipeline.unet(input, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)            
            all_loss = self.accelerator.gather_for_metrics(loss).mean()
            metrics["val_loss"] += all_loss
            #free up memory
            del input, noise, timesteps, noise_pred, loss, all_loss
            torch.cuda.empty_cache()
                
            # get batch
            clean_images = batch["gt_image"]
            masks = batch["mask"]

            images = self._start_pipeline( 
                clean_images,
                masks, 
                parameters
            )

            # transform from B x H x W x C to B x C x H x W 
            images = torch.permute(images, (0, 3, 1, 2))

            all_clean_images = self.accelerator.gather_for_metrics(clean_images)
            all_images = self.accelerator.gather_for_metrics(images)
            all_masks = self.accelerator.gather_for_metrics(masks) 
            new_metrics = EvaluationUtils.calc_metrics(all_clean_images, all_images, all_masks)

            for key, value in new_metrics.items(): 
                metrics[key] += value
        
        for key, value in metrics.items():
            metrics[key] /= self.config.evaluate_num_batches 

        if self.accelerator.is_main_process:
            # log metrics
            EvaluationUtils.log_metrics(self.tb_summary, global_step, metrics)

            # save last batch as sample images
            masked_images = clean_images*(1-masks)
            
            # change range from [-1,1] to [0,1]
            inpainted_images = (inpainted_images+1)/2
            masked_images = (masked_images+1)/2
            clean_images = (clean_images+1)/2
            # change binary image from 0,1 to 0,255
            masks = masks*255

            # save images
            list = [inpainted_images, masked_images, clean_images, masks]
            title_list = ["inpainted_images", "masked_images", "clean_images", "masks"] 
            image_list = [[to_pil_image(x, mode="L") for x in images] for images in list]
            self._save_image(image_list, title_list, os.path.join(self.config.output_dir, "samples_2D"), global_step)