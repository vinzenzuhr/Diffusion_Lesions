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
from torchvision.ops import masks_to_boxes
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

class Evaluation2D(ABC):
    def __init__(self, config, pipeline, dataloader, tb_summary, accelerator, train_env):
        self.config = config
        self.pipeline = pipeline
        self.dataloader = dataloader
        self.tb_summary = tb_summary
        self.accelerator = accelerator
        self.train_env = train_env 
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(self.accelerator.device)

    def _calc_lpip(self, images_1, images_2):

        # create rectangular bounding_boxes
        #all_bounding_boxes = masks_to_boxes(masks.squeeze(dim=1)).to(torch.int32) 
        # returns a [N, 4] tensor containing bounding boxes. The boxes are in (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2.
        #assert all_bounding_boxes.shape[0] == masks.shape[0], "Number of bounding boxes expected to match number of masks."
        
        #calculcate lpips for every image and take average
        lpips = 0
        for i in range(images_1.shape[0]):
            #mask = torch.zeros_like(masks, dtype=torch.bool)
            #mask[i, :, all_bounding_boxes[i][1]:all_bounding_boxes[i][3], all_bounding_boxes[i][0]:all_bounding_boxes[i][2]] = True #TODO: check if correct coordinates
            #width = all_bounding_boxes[i][2] - all_bounding_boxes[i][0]
            #height = all_bounding_boxes[i][3] - all_bounding_boxes[i][1] 
            #img1 = images_1[mask].reshape(1, 1, height, width).expand(-1, 3, -1, -1)
            #img2 = images_2[mask].reshape(1, 1, height, width).expand(-1, 3, -1, -1)
            lpips += self.lpips_metric(images_1[i].unsqueeze(0).expand(-1, 3, -1, -1), images_2[i].unsqueeze(0).expand(-1, 3, -1, -1))
        lpips /= images_1.shape[0]

        return lpips

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
        metric_list = ["ssim_full", "ssim_out", "ssim_in", "mse_full", "mse_out", "mse_in", "psnr_full", "psnr_out", "psnr_in", "val_loss", "lpips"] 
        for metric in metric_list:
            metrics[metric] = 0 
         
        self.progress_bar = tqdm(total=len(self.dataloader), disable=not self.accelerator.is_local_main_process) 
        self.progress_bar.set_description(f"Evaluation 2D") 

        self._reset_seed(self.config.seed)
        for n_iter, batch in enumerate(self.dataloader): 
            if (self.config.evaluate_num_batches != -1) and (n_iter >= self.config.evaluate_num_batches):
                break 
             
            # calc validation loss
            input, noise, timesteps = self.train_env._get_training_input(batch)  
            noise_pred = self.pipeline.unet(input, timesteps, return_dict=False)[0] # kernel dies
            loss = F.mse_loss(noise_pred, noise)          
            all_loss = self.accelerator.gather_for_metrics(loss).mean() 
            metrics["val_loss"] += all_loss 
            #free up memory
            del input, noise, timesteps, noise_pred, loss, all_loss 
            torch.cuda.empty_cache()
            
            # run pipeline. The returned masks can be either existing lesions or the synthetic ones
            images, clean_images, masks = self._start_pipeline( 
                batch,
                parameters
            )

            # transform from B x H x W x C to B x C x H x W 
            images = torch.permute(images, (0, 3, 1, 2))

            # calculate metrics
            all_clean_images = self.accelerator.gather_for_metrics(clean_images)
            all_images = self.accelerator.gather_for_metrics(images)
            all_masks = self.accelerator.gather_for_metrics(masks) 
            metrics["lpips"] += self._calc_lpip(all_clean_images, all_images)
            new_metrics = EvaluationUtils.calc_metrics(all_clean_images, all_images, all_masks)
            for key, value in new_metrics.items(): 
                metrics[key] += value

            self.progress_bar.update(1)

        
        # calculate average metrics
        for key, value in metrics.items():
            metrics[key] /= self.config.evaluate_num_batches 

        if self.accelerator.is_main_process:
            # log metrics
            EvaluationUtils.log_metrics(self.tb_summary, global_step, metrics)

            # save last batch as sample images
            list, title_list = self._get_image_lists(images, clean_images, masks, batch)
            image_list = [[to_pil_image(x, mode="L") for x in images] for images in list]
            self._save_image(image_list, title_list, os.path.join(self.config.output_dir, "samples_2D"), global_step)

            
            

    