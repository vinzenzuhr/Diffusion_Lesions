from Evaluation2D import Evaluation2D
import torch
import numpy as np
import matplotlib.pyplot as plt

class Evaluation2DSynthesis(Evaluation2D):
    def __init__(self, config, pipeline, dataloader, tb_summary, accelerator, train_env):
        assert type(pipeline).__name__ == "DDIMGuidedPipeline", "Pipeline must be of type DDIMGuidedPipeline"
        super().__init__(config, pipeline, dataloader, tb_summary, accelerator, train_env)
    
    def _add_lesions_from_other_lesions(self, clean_images, masks_existing, masks_synthesis): 
        #add new lesions with mean intensity value from existing lesions
        lesion_intensity = clean_images[masks_existing.to(torch.bool)].mean()

        images_with_lesions = clean_images.clone()
        images_with_lesions[masks_synthesis.to(torch.bool)] = lesion_intensity
        return images_with_lesions

    def _add_lesions_from_median_intensity(self, clean_images, masks):
        # median lesion intensity calculcated from the training data
        lesion_intensity = -0.5492

        images_with_lesions = clean_images.clone()
        images_with_lesions[masks.to(torch.bool)] = lesion_intensity
        return images_with_lesions

    def _start_pipeline(self, clean_images, masks, parameters={}):  
        #add coarse lesions
        images_with_lesions = self._add_lesions_from_median_intensity(clean_images, masks)
        print(2) 
        #run it through network
        synthesized_images = self.pipeline(
            images_with_lesions,
            timestep=25,
            generator=torch.cuda.manual_seed_all(self.config.seed),
            output_type=np.array,
            num_inference_steps = self.config.num_inference_steps,
            **parameters
        ).images
        print(3)
        synthesized_images = torch.from_numpy(synthesized_images).to(clean_images.device) 
        return synthesized_images
     