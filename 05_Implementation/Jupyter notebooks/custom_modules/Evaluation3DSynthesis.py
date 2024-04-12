from Evaluation3D import Evaluation3D
import torch
import numpy as np
import math

class Evaluation3DSynthesis(Evaluation3D):
    def __init__(self, config, pipeline, dataloader, tb_summary, accelerator):
        super().__init__(config, pipeline, dataloader, tb_summary, accelerator) 

    def _start_pipeline(self, batch, sample_idx, parameters={}):
        clean_images = batch["gt_image"][sample_idx] #torch.Size([1, 256, 256, 256])
        synthesis_masks = batch["synthesis"][sample_idx]  #torch.Size([1, 256, 256, 256])

        if self.config.add_lesion_technique == "mean_intensity":
            lesion_intensity = -0.5492
        elif self.config.add_lesion_technique == "other_lesions":
            #calculcate mean intensity value of the lesion
            masks = batch["mask"][sample_idx]  #torch.Size([1, 256, 256, 256])
            lesion_intensity = clean_images[masks.to(torch.bool)].mean()
            print("mean lesion intensity: ", lesion_intensity)
        else:
            raise ValueError("config.add_lesion_technique must be either 'mean_intensity' or 'other_lesions'")

        #add new lesions with mean intensity value from existing lesions 
        images_with_lesions = clean_images.clone()
        images_with_lesions[synthesis_masks.to(torch.bool)] = lesion_intensity
    
        #get slices which have to be modified
        slice_indices = []
        for slice_idx in torch.arange(clean_images.shape[2]):
            if (synthesis_masks[:, :, slice_idx, :]).any():
                slice_indices.append(slice_idx.unsqueeze(0)) 
        slice_indices = torch.cat(slice_indices, 0)
        
    
        #create chunks of slices which have to be modified along the horizontal section 
        images_with_lesions = images_with_lesions.permute(2,0,1,3)
        chunks = torch.chunk(images_with_lesions[slice_indices, :, :, :], math.ceil(images_with_lesions.shape[0]/self.config.eval_batch_size), dim=0)

        #modify all slices
        images = [] 
        for chunk_images in chunks: 

            new_images = self.pipeline(
                chunk_images,
                timestep=self.config.intermediate_timestep,
                generator=torch.cuda.manual_seed_all(self.config.seed), 
                num_inference_steps = self.config.num_inference_steps,
                **parameters
            ).images
            #new_images = torch.from_numpy(new_images).to(clean_images.device) 
            images.append(new_images)
        images = torch.cat(images, dim=0)

        images = images.permute(1, 2, 0, 3) 
                
        return images, clean_images, slice_indices, synthesis_masks