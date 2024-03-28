from DatasetMRI import DatasetMRI
import math
import numpy as np
import os
import torch

class Evaluation3D:
    def __init__(self, config, pipeline, dataloader, tb_summary):
        self.config = config
        self.pipeline = pipeline
        self.dataloader = dataloader
        self.tb_summary = tb_summary
    
    def evaluate(self, epoch):
        for batch in self.dataloader:
            # go through sample in batch
            for sample_idx in torch.arange(batch["gt_image"].shape[0]):
                clean_images = batch["gt_image"][sample_idx] #torch.Size([1, 256, 256, 256])
                masks = batch["mask"][sample_idx]  #torch.Size([1, 256, 256, 256])
                max_v = batch["max_v"][sample_idx]
                idx = batch["idx"][sample_idx]
                name = batch["name"][sample_idx]
                voided_images = clean_images*masks
            
                #get slices which have to be inpainted
                slice_indices = []
                for slice_idx in torch.arange(voided_images.shape[2]):
                    if (1-masks[:, :, slice_idx, :]).any():
                        slice_indices.append(slice_idx.unsqueeze(0)) 
                slice_indices = torch.cat(slice_indices, 0)
            
                #create chunks of slices which have to be inpainted
                stacked_void_mask = torch.stack((voided_images[:, :, slice_indices, :], masks[:, :, slice_indices, :]), dim=0)
                stacked_void_mask = stacked_void_mask.permute(0, 3, 1, 2, 4) 
                chunks = torch.chunk(stacked_void_mask, math.ceil(stacked_void_mask.shape[1]/self.config.eval_batch_size), dim=1)
                
                #inpaint all slices
                inpainted_images = [] 
                for chunk in chunks:
                    chunk_voided_images = chunk[0]
                    chunk_masks = chunk[1]

                    size = chunk_masks.shape[0]
                    
                    images = self.pipeline(
                        chunk_voided_images,
                        chunk_masks,
                        batch_size=size,
                        generator=torch.cuda.manual_seed_all(self.config.seed),
                        output_type=np.array,
                        num_inference_steps = self.config.num_inference_steps
                    ).images
                    inpainted_images.append(torch.from_numpy(images))
                inpainted_images = torch.cat(inpainted_images, dim=0)
                inpainted_images = inpainted_images.permute(3, 1, 0, 2)
                inpainted_images = inpainted_images.to(clean_images.device)
            
                #overwrite the original 3D image with the inpainted 2D slices
                clean_images[:, :, slice_indices, :] = inpainted_images
            
                #postprocess and save image as nifti file
                clean_images = DatasetMRI.postprocess(clean_images, max_v)
                save_dir = os.path.join(self.config.output_dir, f"samples_3D/{name}_{epoch:04d}")
                os.makedirs(save_dir, exist_ok=True)
                DatasetMRI.save(clean_images, f"{save_dir}/T1.nii.gz", **self.dataloader.dataset.get_metadata(int(idx)))