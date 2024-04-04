from abc import ABC, abstractmethod
import EvaluationUtils
from DatasetMRI import DatasetMRI
import math
import numpy as np
import os
import torch

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

        print("Start 3D evaluation")
        for batch in self.dataloader:
            # go through sample in batch
            for sample_idx in torch.arange(batch["gt_image"].shape[0]):
                clean_images = batch["gt_image"][sample_idx] #torch.Size([1, 256, 256, 256])
                masks = batch["mask"][sample_idx]  #torch.Size([1, 256, 256, 256])
                max_v = batch["max_v"][sample_idx]
                idx = batch["idx"][sample_idx]
                name = batch["name"][sample_idx]
            
                #get slices which have to be modified
                slice_indices = []
                for slice_idx in torch.arange(clean_images.shape[2]):
                    if (masks[:, :, slice_idx, :]).any():
                        slice_indices.append(slice_idx.unsqueeze(0)) 
                slice_indices = torch.cat(slice_indices, 0)
            
                #create chunks of slices which have to be modified
                stacked_images = torch.stack((clean_images[:, :, slice_indices, :], masks[:, :, slice_indices, :]), dim=0)
                stacked_images = stacked_images.permute(0, 3, 1, 2, 4) 
                chunks = torch.chunk(stacked_images, math.ceil(stacked_images.shape[1]/self.config.eval_batch_size), dim=1)
                
                #modify all slices
                images = [] 
                for chunk in chunks:
                    chunk_images = chunk[0]
                    chunk_masks = chunk[1]

                    new_images = self._start_pipeline(chunk_images, chunk_masks, parameters)

                    images.append(new_images)
                images = torch.cat(images, dim=0)
                images = images.permute(3, 1, 0, 2) 
            
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
                final_3d_images = DatasetMRI.postprocess(final_3d_images, max_v)  
                save_dir = os.path.join(self.config.output_dir, f"samples_3D/{name}") 
                os.makedirs(save_dir, exist_ok=True)
                DatasetMRI.save(final_3d_images, f"{save_dir}/T1.nii.gz", **self.dataloader.dataset.get_metadata(int(idx)))
        
        # calculcate mean of metrics and log them
        for key, value in metrics.items():
            metrics[key] /= num_iterations
        if self.accelerator.is_main_process: 
            EvaluationUtils.log_metrics(self.tb_summary, global_step, metrics)

        print("3D evaluation finished")