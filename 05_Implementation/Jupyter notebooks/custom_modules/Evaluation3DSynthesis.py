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
        masks = batch["mask"][sample_idx]  #torch.Size([1, 256, 256, 256])
        if self.config.add_lesion_technique == "mean_intensity":
            lesion_intensity = -0.5492 
        elif self.config.add_lesion_technique == "other_lesions_1stQuantile":
            # use first quantile of lesion intensity as new lesion intensity
            lesion_intensity = clean_images[masks].quantile(0.25)
            print("1st quantile lesion intensity: ", lesion_intensity)
        elif self.config.add_lesion_technique == "other_lesions_mean":
            # use mean of lesion intensity as new lesion intensity
            lesion_intensity = clean_images[masks].mean()
            print("mean lesion intensity: ", lesion_intensity)
        elif self.config.add_lesion_technique == "other_lesions_median":
            # use mean of lesion intensity as new lesion intensity
            lesion_intensity = clean_images[masks].median()
            print("median lesion intensity: ", lesion_intensity)
        elif self.config.add_lesion_technique == "other_lesions_3rdQuantile":
            # use 3rd quantile of lesion intensity as new lesion intensity
            lesion_intensity = clean_images[masks].quantile(0.75)
            print("3rd quantile lesion intensity: ", lesion_intensity)
        else:
            raise ValueError("config.add_lesion_technique must be either 'mean_intensity' or 'other_lesions'")

        #add new lesions with mean intensity value from existing lesions 
        images_with_lesions = clean_images.clone()
        images_with_lesions[synthesis_masks.to(torch.bool)] = lesion_intensity
    
        if self.config.num_samples_per_batch>1:
            raise ValueError("num_samples_per_batch must be implmented for Evaluation3DSynthesis")

        # get slices which have to be modified.  Make sure that there are at least num_samples slices in a package, which are located next to each other.
        position_in_package = 0
        slice_indices = []
        for slice_idx in torch.arange(clean_images.shape[2]):
            if (synthesis_masks[:, :, slice_idx, :]).any() or position_in_package>0:
                slice_indices.append(slice_idx.unsqueeze(0)) 
                position_in_package += 1
                if position_in_package == self.config.num_samples_per_batch:
                    position_in_package = 0  
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
                generator=torch.Generator().manual_seed(self.config.seed), 
                num_inference_steps = self.config.num_inference_steps,
                **parameters
            ).images
            #new_images = torch.from_numpy(new_images).to(clean_images.device) 
            images.append(new_images)
        images = torch.cat(images, dim=0)

        images = images.permute(1, 2, 0, 3) 
                
        return images, clean_images, slice_indices, synthesis_masks