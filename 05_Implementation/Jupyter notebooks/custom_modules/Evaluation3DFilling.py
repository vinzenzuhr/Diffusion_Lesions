from Evaluation3D import Evaluation3D
import torch
import numpy as np
import math

class Evaluation3DFilling(Evaluation3D):
    def __init__(self, config, pipeline, dataloader, tb_summary, accelerator):
        super().__init__(config, pipeline, dataloader, tb_summary, accelerator)

    def _start_pipeline(self, batch, sample_idx, parameters={}):

        clean_images = batch["gt_image"][sample_idx] #torch.Size([1, 256, 256, 256])
        masks = batch["mask"][sample_idx]  #torch.Size([1, 256, 256, 256])
    
        #get slices which have to be modified
        slice_indices = []
        for slice_idx in torch.arange(clean_images.shape[2]):
            if (masks[:, :, slice_idx, :]).any():
                slice_indices.append(slice_idx.unsqueeze(0)) 
        slice_indices = torch.cat(slice_indices, 0)
    
        #create chunks of slices which have to be modified along the horizontal section 
        stacked_images = torch.stack((clean_images[:, :, slice_indices, :], masks[:, :, slice_indices, :]), dim=0)
        stacked_images = stacked_images.permute(0, 3, 1, 2, 4) 
        chunks = torch.chunk(stacked_images, math.ceil(stacked_images.shape[1]/self.config.eval_batch_size), dim=1)
        
        #modify all slices
        images = [] 
        for chunk in chunks:
            chunk_images = chunk[0]
            chunk_masks = chunk[1]
            chunk_voided_images = chunk_images*(1-chunk_masks)

            new_images = self.pipeline(
                chunk_voided_images,
                chunk_masks,
                generator=torch.cuda.manual_seed_all(self.config.seed),
                output_type=np.array,
                num_inference_steps = self.config.num_inference_steps,
                **parameters
            ).images
            new_images = torch.from_numpy(new_images).to(clean_images.device) 
            images.append(new_images)
        images = torch.cat(images, dim=0)

        return images, clean_images, slice_indices, masks