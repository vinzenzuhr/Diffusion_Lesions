from Evaluation2D import Evaluation2D
import torch
import numpy as np
import matplotlib.pyplot as plt 

class Evaluation2DSynthesis(Evaluation2D):
    def __init__(self, config, pipeline, dataloader, tb_summary, accelerator, _get_training_input):
        assert type(pipeline).__name__ == "DDIMGuidedPipeline", "Pipeline must be of type DDIMGuidedPipeline"
        super().__init__(config, pipeline, dataloader, tb_summary, accelerator, _get_training_input)

    def _add_coarse_lesions(self, clean_images, batch):
        synthesis_masks = batch["synthesis"] 
        masks = batch["mask"].to(torch.bool)
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
        images_with_lesions = clean_images.clone()
        images_with_lesions[synthesis_masks.to(torch.bool)] = lesion_intensity 
        return images_with_lesions, synthesis_masks

    def _get_image_lists(self, images, clean_images, masks, batch):
        images_with_lesions, synthesis_masks = self._add_coarse_lesions(clean_images, batch)

        masked_images = clean_images*(1-synthesis_masks)
        
        # change range from [-1,1] to [0,1]
        images = (images+1)/2
        masked_images = (masked_images+1)/2
        clean_images = (clean_images+1)/2
        images_with_lesions = (images_with_lesions+1)/2
        # change binary image from 0,1 to 0,255
        synthesis_masks = synthesis_masks*255 

        list = [images, masked_images, clean_images, synthesis_masks, images_with_lesions]
        title_list = ["images", "masked_images", "clean_images", "masks", "images_with_lesions"] 
        return list, title_list

    def _start_pipeline(self, batch, parameters={}):
        clean_images = batch["gt_image"]
          
        #add coarse lesions
        images_with_lesions, synthesis_masks = self._add_coarse_lesions(clean_images, batch)

        #run it through network        
        synthesized_images = self.pipeline(
            images_with_lesions,
            timestep=self.config.intermediate_timestep,
            generator=torch.Generator().manual_seed(self.config.seed), 
            num_inference_steps = self.config.num_inference_steps,
            **parameters
        ).images     
        #synthesized_images = torch.from_numpy(synthesized_images).to(clean_images.device)  
        return synthesized_images, clean_images, synthesis_masks
     