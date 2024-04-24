from abc import ABC, abstractmethod
import EvaluationUtils
from DatasetMRI import DatasetMRI
import math
import numpy as np
import os
import torch
from tqdm.auto import tqdm

class Evaluation3D(ABC):
    def __init__(self, config, pipeline, dataloader, tb_summary, accelerator):
        self.config = config
        self.pipeline = pipeline
        self.dataloader = dataloader
        self.tb_summary = tb_summary
        self.accelerator = accelerator

        # create folder for segmentation algorithm afterwards
        segmentation_dir = os.path.join(config.output_dir, "segmentations_3D")
        os.makedirs(segmentation_dir, exist_ok=True) 

    @abstractmethod
    def _start_pipeline(self, clean_images, masks, parameters):
        pass
    
    def evaluate(self, global_step, parameters={}):
        #initialize metrics 
        metrics = dict()
        metric_list = ["ssim_full", "ssim_out", "ssim_in", "mse_full", "mse_out", "mse_in", "psnr_full", "psnr_out", "psnr_in"] 
        for metric in metric_list:
            metrics[metric] = 0
        num_iterations = 0

        
        self.progress_bar = tqdm(total=len(self.dataloader), disable=not self.accelerator.is_local_main_process) 
        self.progress_bar.set_description(f"Evaluation 3D") 
 
        print("Start 3D evaluation")
        for n_iter, batch in enumerate(self.dataloader): 
            # go through sample in batch
            for sample_idx in torch.arange(batch["gt_image"].shape[0]):
                
                idx = batch["idx"][sample_idx]
                name = batch["name"][sample_idx]
                proc_info = batch["proc_info"][sample_idx]

                images, clean_images, slice_indices, masks = self._start_pipeline(batch, sample_idx, parameters)    
            
                #overwrite the original 3D image with the modified 2D slices
                final_3d_images = torch.clone(clean_images.detach()) 
                final_3d_images[:, :, slice_indices, :] = images

                #calculate metrics
                all_clean_images = self.accelerator.gather_for_metrics(clean_images)
                all_3d_images = self.accelerator.gather_for_metrics(final_3d_images)
                all_masks = self.accelerator.gather_for_metrics(masks)
                new_metrics = EvaluationUtils.calc_metrics(all_clean_images, all_3d_images, all_masks)

                for key, value in new_metrics.items(): 
                    metrics[key] += value 
                num_iterations += 1
            
                #postprocess and save image as nifti file
                final_3d_images = DatasetMRI.postprocess(final_3d_images.squeeze(), *proc_info, self.dataloader.dataset.get_metadata(int(idx)))  
                save_dir = os.path.join(self.config.output_dir, f"samples_3D/{name}") 
                os.makedirs(save_dir, exist_ok=True)
                DatasetMRI.save(final_3d_images, f"{save_dir}/T1.nii.gz")

            self.progress_bar.update(1)
            
            if (self.config.evaluate_num_batches != -1) and (n_iter >= self.config.evaluate_num_batches-1):
                break 
        
        # calculcate mean of metrics and log them
        for key, value in metrics.items():
            metrics[key] /= num_iterations

        #rename metrics to 3D metrics
        dim3_metrics = dict()
        for key, value in metrics.items():
            dim3_metrics[f"{key}_3D"] = value
        if self.accelerator.is_main_process: 
            EvaluationUtils.log_metrics(self.tb_summary, global_step, dim3_metrics, self.config)

        print("3D evaluation finished")