from Evaluation2D import Evaluation2D
import torch
import numpy as np
import matplotlib.pyplot as plt

class Evaluation2DSynthesis(Evaluation2D):
    def __init__(self, config, pipeline, dataloader, tb_summary, accelerator, train_env):
        assert type(pipeline).__name__ == "DDIMGuidedPipeline", "Pipeline must be of type DDIMGuidedPipeline"
        super().__init__(config, pipeline, dataloader, tb_summary, accelerator, train_env)

    def _add_lesions(self, clean_images, masks):
        #add lesions
        images_with_lesions = clean_images.clone()
        images_with_lesions[masks] = -1
        return images_with_lesions
    
    def _add_lesions_from_gm(self, clean_images, masks, segmentation):
        segmentation = segmentation.unsqueeze(0) 
        assert clean_images.shape == segmentation.shape, "clean_images and segmentation must have the same shape"

        #get mean value of gray matter
        gm_labels = torch.cat([torch.arange(1000,1040), torch.arange(2000,2040)])
        gm_mask = torch.isin(segmentation, gm_labels) 
        gm_mean = clean_images[gm_mask].mean()
        gm_std = clean_images[gm_mask].std()

        #add white matter lesions with mean value of gray matter minus three times the standard deviation
        print("gm_mean: ", gm_mean)
        print("gm_std: ", gm_std)
        print("new value: ", gm_mean - 3*gm_std)
        images_with_lesions = clean_images.clone()
        images_with_lesions[masks.to(torch.bool)] = gm_mean - 3*gm_std
        return images_with_lesions

    def _start_pipeline(self, clean_images, masks, segmentation, parameters={}):  
        #add coarse lesions
        images_with_lesions = self._add_lesions_from_gm(clean_images, masks, segmentation)
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