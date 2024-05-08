from custom_modules import Evaluation2D

import torch
import numpy as np 

class Evaluation2DFilling(Evaluation2D):
    def __init__(self, config, dataloader, tb_summary, accelerator):
        super().__init__(config, dataloader, tb_summary, accelerator) 
    
    def _get_image_lists(self, images, clean_images, masks, batch):
        # save last batch as sample images
        masked_images = clean_images*(1-masks)
        
        # change range from [-1,1] to [0,1]
        images = (images+1)/2
        masked_images = (masked_images+1)/2
        clean_images = (clean_images+1)/2  

        list = [images, masked_images, clean_images, masks]
        title_list = ["images", "masked_images", "clean_images", "masks"] 
        return list, title_list

    def _start_pipeline(self, pipeline, batch, generator, parameters={}): 
        clean_images = batch["gt_image"]
        masks = batch["mask"] 
        voided_images = clean_images*(1-masks)
         
        inpainted_images = pipeline(
            voided_images,
            masks,
            generator=generator, 
            num_inference_steps = self.config.num_inference_steps,
            **parameters
        ).images 
        
        #inpainted_images = torch.from_numpy(inpainted_images).to(clean_images.device) 
        return inpainted_images, clean_images, masks
    
