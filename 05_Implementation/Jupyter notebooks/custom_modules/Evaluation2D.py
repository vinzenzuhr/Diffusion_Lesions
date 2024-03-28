#setup evaluation
from diffusers.utils import make_image_grid
import numpy as np
import os
from pathlib import Path
import PIL 
import random
from skimage.metrics import structural_similarity, mean_squared_error
import torch
from torcheval.metrics import PeakSignalNoiseRatio
from torchvision.transforms.functional import to_pil_image

class Evaluation2D:
    def __init__(self, config, pipeline, dataloader, tb_summary):
        self.config = config
        self.pipeline = pipeline
        self.dataloader = dataloader
        self.tb_summary = tb_summary

    def _calc_metrics(self, image1, image2):
        batch_size = image1.shape[0]

        # PSNR metric
        metric = PeakSignalNoiseRatio(device=image1.device)
        metric.update(image1, image2)
        PSNR = metric.compute().item()

        # SSIM metric
        mssim_sum=0
        for idx in range(batch_size):
            mssim = structural_similarity(
                image1[idx].detach().cpu().numpy(),
                image2[idx].detach().cpu().numpy(),
                channel_axis=0,
                data_range=2
            )
            mssim_sum += mssim
        SSIM = mssim_sum / batch_size

        # MSE metric
        mse_sum=0
        for idx in range(batch_size):
            mse = mean_squared_error(
                image1[idx].detach().cpu().numpy(),
                image2[idx].detach().cpu().numpy(), 
            )
            mse_sum += mse
        MSE = mse_sum / batch_size

        return PSNR, SSIM, MSE

    def _log_metrics(self, tb_summary, global_step, PSNR_mean, SSIM_mean, MSE_mean):
        tb_summary.add_scalar("PSNR_global", PSNR_mean, global_step) 
        tb_summary.add_scalar("SSIM_global", SSIM_mean, global_step)
        tb_summary.add_scalar("MSE_global", MSE_mean, global_step)
        print("SSIM_global: ", SSIM_mean)
        print("PSNR_global: ", PSNR_mean)
        print("MSE_global: ", MSE_mean)
        print("global_step: ", global_step)

    def _reset_seed(self, seed): 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        return

    def _save_image(self, images: list[list[PIL.Image]], titles: list[str], path: Path, epoch: int):
        os.makedirs(path, exist_ok=True)
        for image_list, title in zip(images, titles): 
            image_grid = make_image_grid(image_list, rows=int(len(image_list)**0.5), cols=int(len(image_list)**0.5))
            image_grid.save(f"{path}/{title}_{epoch:04d}.png")
        print("image saved")   

    def evaluate(self, epoch, global_step):
        #initialize metrics
        PSNR_mean = 0
        SSIM_mean = 0  
        MSE_mean = 0

        self._reset_seed(self.config.seed)
        for n_iter, batch in enumerate(self.dataloader): 
            if n_iter >= self.config.evaluate_num_batches:
                break
                
            # get batch
            clean_images = batch["gt_image"]
            masks = batch["mask"]
            voided_images = clean_images*masks 

            # run them through pipeline
            inpainted_images = self.pipeline(
                voided_images,
                masks,
                batch_size=masks.shape[0],
                generator=torch.cuda.manual_seed_all(self.config.seed),
                output_type=np.array,
                num_inference_steps = self.config.num_inference_steps
            ).images
            inpainted_images = torch.from_numpy(inpainted_images).to(clean_images.device)
            
            # transform from B x H x W x C to B x C x H x W 
            inpainted_images = torch.permute(inpainted_images, (0, 3, 1, 2))

            PSNR, SSIM, MSE = self._calc_metrics(clean_images, inpainted_images)
            PSNR_mean += PSNR
            SSIM_mean += SSIM 
            MSE_mean += MSE
            
        # calculcate mean of metrics
        PSNR_mean /= len(self.dataloader)
        SSIM_mean /= len(self.dataloader)
        MSE_mean /= len(self.dataloader)

        

        self._log_metrics(self.tb_summary, global_step, PSNR_mean, SSIM_mean, MSE_mean)

        # save last batch as sample images
        if (epoch) % self.config.evaluate_save_img_epochs == 0 or epoch == self.config.num_epochs - 1:
            # change range from [-1,1] to [0,1]
            voided_images = (voided_images+1)/2
            clean_images = (clean_images+1)/2
            # change binary image from 0,1 to 0,255
            masks = masks*255

            # save images
            list = [inpainted_images, voided_images, clean_images, masks]
            title_list = ["inpainted_images", "voided_images", "clean_images", "masks"] 
            image_list = [[to_pil_image(x, mode="L") for x in images] for images in list]
            self._save_image(image_list, title_list, os.path.join(self.config.output_dir, "samples_2D"), epoch)