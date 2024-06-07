from custom_modules import Evaluation2D, EvaluationUtils

import torch
import numpy as np
import matplotlib.pyplot as plt 

class Evaluation2DSynthesis(Evaluation2D):
    def __init__(self, config, eval_dataloader, train_dataloader, tb_summary, accelerator):
        super().__init__(config, eval_dataloader, train_dataloader, tb_summary, accelerator)

    def _add_coarse_lesions(self, clean_images, batch):
        synthesis_masks = batch["synthesis"] 
        masks = batch["mask"].to(torch.bool) 
        lesion_intensity = EvaluationUtils.get_lesion_intensity(
            self.config.add_lesion_technique, 
            clean_images[masks], 
            self.config.add_lesion_mean_intensity)
        
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

        list = [images, masked_images, clean_images, synthesis_masks, images_with_lesions]
        title_list = ["images", "masked_images", "clean_images", "synthesis_masks", "images_with_lesions"] 
        return list, title_list

    def _start_pipeline(self, pipeline, batch, generator, parameters={}):
        
        assert (type(pipeline).__name__ == "GuidedRePaintPipeline" 
                or type(pipeline).__name__ == "GuidedPipelineUnconditional"
                or type(pipeline).__name__ == "GuidedPipelineConditional") , \
            "Pipeline must be of type DDIMGuidedPipeline or GuidedRePaintPipeline"

        clean_images = batch["gt_image"]
          
        #add coarse lesions
        images_with_lesions, synthesis_masks = self._add_coarse_lesions(clean_images, batch)

        #run it through network        
        synthesized_images = pipeline(
            images_with_lesions,
            synthesis_masks,
            timestep=self.config.intermediate_timestep,
            generator=generator, 
            num_inference_steps = self.config.num_inference_steps,
            **parameters
        ).images     
        #synthesized_images = torch.from_numpy(synthesized_images).to(clean_images.device)  
        return synthesized_images, clean_images, synthesis_masks
     